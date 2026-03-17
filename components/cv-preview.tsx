"use client"

import { Download, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Fragment, useState } from "react"

interface CVSection {
  heading: string
  content: string
}

interface ContactItem {
  display: string
  url?: string
}

interface BulletItem {
  level: "primary" | "secondary"
  text: string
  label?: string
  detail?: string
}

interface TextUrlMatch {
  text: string
  start: number
  end: number
  url: string
}

const URL_REGEX = /(?:https?:\/\/[^\s)]+|www\.[^\s)]+|(?:github|linkedin|youtube)\.com\/[^\s)]+|youtu\.be\/[^\s)]+)/gi

interface CVData {
  fullName: string
  title: string
  email: string
  phone: string
  location: string
  summary: string
  sections: CVSection[]
}

interface CVPreviewProps {
  data: CVData | null
  isGenerating: boolean
  streamingText: string
}

export function CVPreview({ data, isGenerating, streamingText }: CVPreviewProps) {
  const [isDownloading, setIsDownloading] = useState(false)
  const { contactProfiles, remainingSections } = extractOnlineProfiles(data?.sections ?? [])
  const profileEntries = contactProfiles.map(parseOnlineProfile)
  const baseContactItems: ContactItem[] = [data?.email, data?.phone, data?.location]
    .filter((value): value is string => Boolean(value))
    .map((value) => ({
      display: value,
    }))
  const contactItems: ContactItem[] = [...baseContactItems, ...profileEntries]

  const handleDownload = async () => {
    if (!data) return
    setIsDownloading(true)
    try {
      const { PDFDocument, rgb, StandardFonts, PDFName, PDFString } = await import("pdf-lib")

      const pdfDoc = await PDFDocument.create()
      const helvetica = await pdfDoc.embedFont(StandardFonts.Helvetica)
      const helveticaBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold)

      const pageWidth = 595.28
      const pageHeight = 841.89
      const margin = 50
      const contentWidth = pageWidth - margin * 2

      let page = pdfDoc.addPage([pageWidth, pageHeight])
      let y = pageHeight - margin

      const black = rgb(0, 0, 0)
      const darkBlue = black
      const gray = black

      const drawCenteredText = (
        text: string,
        size: number,
        font: typeof helvetica,
        color: ReturnType<typeof rgb>
      ) => {
        const textWidth = font.widthOfTextAtSize(text, size)
        const x = margin + (contentWidth - textWidth) / 2
        page.drawText(text, {
          x: Math.max(margin, x),
          y,
          size,
          font,
          color,
        })

        return {
          x: Math.max(margin, x),
          textWidth,
        }
      }

      const drawTextWithLinks = (
        text: string,
        x: number,
        yPos: number,
        font: typeof helvetica,
        size: number,
        color: ReturnType<typeof rgb>
      ) => {
        page.drawText(text, {
          x,
          y: yPos,
          size,
          font,
          color,
        })

        for (const match of extractUrlMatches(text)) {
          const prefixText = text.slice(0, match.start)
          const matchText = text.slice(match.start, match.end)
          const matchX = x + font.widthOfTextAtSize(prefixText, size)
          const matchWidth = font.widthOfTextAtSize(matchText, size)
          addPdfLinkAnnotation(match.url, matchX, yPos - 1, matchWidth, size + 2)
        }
      }

      const addPdfLinkAnnotation = (url: string, x: number, yPos: number, width: number, height: number) => {
        const linkAnnotation = pdfDoc.context.obj({
          Type: PDFName.of("Annot"),
          Subtype: PDFName.of("Link"),
          Rect: [x, yPos, x + width, yPos + height],
          Border: [0, 0, 0],
          A: pdfDoc.context.obj({
            Type: PDFName.of("Action"),
            S: PDFName.of("URI"),
            URI: PDFString.of(url),
          }),
        })

        const linkAnnotationRef = pdfDoc.context.register(linkAnnotation)
        page.node.addAnnot(linkAnnotationRef)
      }

      const addNewPageIfNeeded = (requiredSpace: number) => {
        if (y - requiredSpace < margin) {
          page = pdfDoc.addPage([pageWidth, pageHeight])
          y = pageHeight - margin
        }
      }

      const drawCenteredContactLine = (items: ContactItem[]) => {
        if (items.length === 0) return false

        const separator = "  |  "
        const fontSize = 9
        const separatorWidth = helvetica.widthOfTextAtSize(separator, fontSize)
        const itemWidths = items.map((item) => helvetica.widthOfTextAtSize(item.display, fontSize))

        const lines: Array<Array<{ item: ContactItem; width: number; withSeparator: boolean }>> = []
        let currentLine: Array<{ item: ContactItem; width: number; withSeparator: boolean }> = []
        let currentLineWidth = 0

        items.forEach((item, index) => {
          const itemWidth = itemWidths[index]
          const hasPrefixSeparator = currentLine.length > 0
          const neededWidth = itemWidth + (hasPrefixSeparator ? separatorWidth : 0)

          if (currentLineWidth + neededWidth > contentWidth && currentLine.length > 0) {
            lines.push(currentLine)
            currentLine = [{ item, width: itemWidth, withSeparator: false }]
            currentLineWidth = itemWidth
            return
          }

          currentLine.push({ item, width: itemWidth, withSeparator: hasPrefixSeparator })
          currentLineWidth += neededWidth
        })

        if (currentLine.length > 0) {
          lines.push(currentLine)
        }

        for (const line of lines) {
          const lineText = line
            .map((entry, index) => (index === 0 ? entry.item.display : `${separator}${entry.item.display}`))
            .join("")
          const { x } = drawCenteredText(lineText, fontSize, helvetica, gray)

          let currentX = x
          for (const entry of line) {
            if (entry.withSeparator) {
              currentX += separatorWidth
            }

            if (entry.item.url) {
              addPdfLinkAnnotation(entry.item.url, currentX, y - 1, entry.width, fontSize + 2)
            }

            currentX += entry.width
          }

          y -= fontSize + 3
        }

        return true
      }

      // Name
      drawCenteredText(data.fullName, 24, helveticaBold, darkBlue)
      y -= 28

      // Title
      if (data.title) {
        drawCenteredText(data.title, 12, helveticaBold, black)
        y -= 20
      }

      // Contact info
      if (drawCenteredContactLine(contactItems)) {
        y -= 8
      }

      y -= 12

      // Summary
      if (data.summary) {
        addNewPageIfNeeded(34)
        page.drawText("PROFILE", {
          x: margin,
          y,
          size: 11,
          font: helveticaBold,
          color: black,
        })
        y -= 6

        page.drawLine({
          start: { x: margin, y },
          end: { x: pageWidth - margin, y },
          thickness: 1,
          color: black,
        })
        y -= 14

        const summaryLines = wrapText(data.summary, helvetica, 10, contentWidth)
        for (const line of summaryLines) {
          addNewPageIfNeeded(14)
          page.drawText(line, {
            x: margin,
            y,
            size: 10,
            font: helvetica,
            color: black,
          })
          y -= 14
        }
        y -= 10
      }

      // Sections
      for (const section of remainingSections) {
        addNewPageIfNeeded(40)

        const sectionHeading =
          section.heading.trim().toLowerCase() === "experience"
            ? "Work Experience"
            : section.heading
        const isPrimaryBoldSection = shouldBoldPrimaryForSection(sectionHeading)

        page.drawText(sectionHeading.toUpperCase(), {
          x: margin,
          y,
          size: 11,
          font: helveticaBold,
          color: darkBlue,
        })
        y -= 6

        page.drawLine({
          start: { x: margin, y },
          end: { x: pageWidth - margin, y },
          thickness: 1,
          color: black,
        })
        y -= 14

        const bulletItems = parseSectionBulletItems(section.heading, section.content)
        for (const item of bulletItems) {
          const fontSize = 10

          if (item.level === "primary") {
            if (item.label && item.detail) {
              const bulletPrefix = "• "
              const labelText = `${bulletPrefix}${item.label}: `
              const labelWidth = helveticaBold.widthOfTextAtSize(labelText, fontSize)
              const firstLineAvailableWidth = Math.max(80, contentWidth - labelWidth)

              const detailWords = item.detail.split(/\s+/).filter(Boolean)
              let firstLineDetail = ""
              let consumedWords = 0

              for (let index = 0; index < detailWords.length; index += 1) {
                const candidate = firstLineDetail ? `${firstLineDetail} ${detailWords[index]}` : detailWords[index]
                if (helvetica.widthOfTextAtSize(candidate, fontSize) <= firstLineAvailableWidth) {
                  firstLineDetail = candidate
                  consumedWords = index + 1
                } else {
                  break
                }
              }

              const remainingDetail = detailWords.slice(consumedWords).join(" ")
              const continuationLines = remainingDetail
                ? wrapText(remainingDetail, helvetica, fontSize, Math.max(80, contentWidth - 12))
                : []

              addNewPageIfNeeded(14)
              drawTextWithLinks(labelText, margin, y, helveticaBold, fontSize, gray)

              if (firstLineDetail) {
                drawTextWithLinks(firstLineDetail, margin + labelWidth, y, helvetica, fontSize, gray)
              }

              y -= 14

              for (const line of continuationLines) {
                addNewPageIfNeeded(14)
                drawTextWithLinks(line, margin + 12, y, helvetica, fontSize, gray)
                y -= 14
              }

              y -= 2
              continue
            }

            const primaryPrefix = "• "
            const { leftText, dateText } = splitDateSuffix(item.text)
            const primaryFont = isPrimaryBoldSection ? helveticaBold : helvetica
            const dateFont = helveticaBold
            const dateWidth = dateText ? dateFont.widthOfTextAtSize(dateText, fontSize) : 0
            const dateGap = dateText ? 12 : 0
            const leftMaxWidth = Math.max(120, contentWidth - dateWidth - dateGap)

            const leftLines = wrapText(`${primaryPrefix}${leftText}`, primaryFont, fontSize, leftMaxWidth)
            const leftStartY = y

            for (const line of leftLines) {
              addNewPageIfNeeded(14)
              drawTextWithLinks(line, margin, y, primaryFont, fontSize, gray)
              y -= 14
            }

            if (dateText) {
              drawTextWithLinks(
                dateText,
                pageWidth - margin - dateWidth,
                leftStartY,
                dateFont,
                fontSize,
                gray
              )
            }

            y -= 2
            continue
          }

          const prefix = "  - "
          const font = helvetica
          const lines = wrapText(`${prefix}${item.text}`, font, fontSize, contentWidth)

          for (const line of lines) {
            addNewPageIfNeeded(14)
            drawTextWithLinks(line, margin, y, font, fontSize, gray)
            y -= 14
          }

        }
        y -= 12
      }

      const pdfBytes = await pdfDoc.save()
      const pdfArrayBuffer = new ArrayBuffer(pdfBytes.byteLength)
      new Uint8Array(pdfArrayBuffer).set(pdfBytes)
      const blob = new Blob([pdfArrayBuffer], { type: "application/pdf" })
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `${data.fullName.replace(/\s+/g, "_")}_CV.pdf`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error("PDF generation error:", error)
    } finally {
      setIsDownloading(false)
    }
  }

  if (isGenerating) {
    return (
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin text-accent" />
          <span className="text-sm font-semibold uppercase tracking-wider text-foreground">
            Generating your CV...
          </span>
        </div>
        {streamingText && (
          <div className="rounded-lg border border-primary/30 bg-card/80 p-6">
            <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-muted-foreground">
              {streamingText}
            </pre>
          </div>
        )}
      </div>
    )
  }

  if (!data) {
    return (
      <div className="flex min-h-[400px] flex-col items-center justify-center gap-4 rounded-lg border border-dashed border-primary/30 bg-card/70 p-8">
        <div className="flex h-16 w-16 items-center justify-center rounded-full border border-dashed border-primary/40 bg-background/70">
          <Download className="h-6 w-6 text-muted-foreground" />
        </div>
        <div className="flex flex-col items-center gap-1 text-center">
          <span className="text-sm font-semibold text-foreground">Your new CV will appear here</span>
          <span className="text-xs text-muted-foreground">
            Upload your old CV and paste a job offer to get started
          </span>
        </div>
      </div>
    )
  }

  return (
    (() => {
      return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold uppercase tracking-wider text-black">
          Your Tailored CV
        </span>
        <Button
          onClick={handleDownload}
          disabled={isDownloading}
          className="bg-primary text-primary-foreground hover:bg-primary/90"
        >
          {isDownloading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Preparing PDF...
            </>
          ) : (
            <>
              <Download className="mr-2 h-4 w-4" />
              Download PDF
            </>
          )}
        </Button>
      </div>

      <div className="overflow-hidden rounded-lg border border-slate-300 bg-white shadow-[0_16px_48px_rgba(2,6,23,0.18)]">
        {/* Header */}
        <div className="border-b border-slate-300 bg-slate-100 p-6">
          <h2 className="text-2xl font-bold text-black">{data.fullName}</h2>
          {data.title && (
            <p className="mt-1 text-sm font-medium text-black">{data.title}</p>
          )}
          <div className="mt-2 text-[11px] leading-relaxed text-black">
            <div className="flex flex-wrap items-center gap-y-1">
              {contactItems.map((item, index) => (
                <span key={`${item.display}-${index}`} className="inline-flex items-center whitespace-nowrap">
                  {index > 0 && <span className="mx-2 text-slate-400">|</span>}
                {item.url ? (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noreferrer"
                    className="underline decoration-slate-500/50 underline-offset-2 hover:text-slate-900"
                  >
                    {item.display}
                  </a>
                ) : (
                  <span>{item.display}</span>
                )}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex flex-col gap-6 bg-white p-6">
          {data.summary && (
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold uppercase tracking-widest text-black">Profile</h3>
                <div className="h-px flex-1 bg-slate-300" />
              </div>
              <p className="text-sm leading-relaxed text-black">{data.summary}</p>
            </div>
          )}

          {remainingSections.map((section, index) => (
            <div key={index} className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold uppercase tracking-widest text-black">
                  {section.heading}
                </h3>
                <div className="h-px flex-1 bg-slate-300" />
              </div>
              <div className="flex flex-col gap-1 text-sm leading-relaxed text-black">
                {parseSectionBulletItems(section.heading, section.content).map((item, itemIndex) => (
                  <div
                    key={`${section.heading}-${itemIndex}`}
                    className={item.level === "primary" ? "flex items-start gap-2" : "ml-5 flex items-start gap-2"}
                  >
                    <span className="w-4 shrink-0 text-center">{item.level === "primary" ? "•" : "-"}</span>
                    {item.label && item.detail ? (
                      <span>
                        <span className="font-semibold">{item.label}</span>
                        <span>: <LinkifiedText text={item.detail} /></span>
                      </span>
                    ) : (
                      <span
                        className={
                          shouldBoldPrimaryForSection(section.heading) && item.level === "primary"
                            ? "font-semibold"
                            : ""
                        }
                      >
                        <LinkifiedText text={item.text} />
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
      )
    })()
  )
}

function extractOnlineProfiles(sections: CVSection[]): {
  contactProfiles: string[]
  remainingSections: CVSection[]
} {
  const contactProfiles: string[] = []
  const remainingSections: CVSection[] = []

  for (const section of sections) {
    const normalizedHeading = section.heading.trim().toLowerCase()
    const isOnlineProfiles =
      normalizedHeading === "online profiles" || normalizedHeading === "online profile"

    if (!isOnlineProfiles) {
      remainingSections.push(section)
      continue
    }

    const lines = section.content
      .split("\n")
      .map((line) => line.trim().replace(/^-+\s*/, ""))
      .filter(Boolean)

    contactProfiles.push(...lines)
  }

  return { contactProfiles, remainingSections }
}

function parseOnlineProfile(profileLine: string): { display: string; url?: string } {
  const trimmed = profileLine.trim()
  if (!trimmed) {
    return { display: profileLine }
  }

  const directUrlMatch = trimmed.match(/https?:\/\/\S+/i)
  if (directUrlMatch) {
    return {
      display: trimmed,
      url: normalizeUrl(directUrlMatch[0]),
    }
  }

  const colonParts = trimmed.split(":")
  if (colonParts.length > 1) {
    const platform = colonParts[0].trim().toLowerCase()
    const value = colonParts.slice(1).join(":").trim()
    const valueUrl = extractUrl(value)
    if (valueUrl) {
      return {
        display: trimmed,
        url: valueUrl,
      }
    }

    const platformUrl = buildPlatformUrl(platform, value)
    if (platformUrl) {
      return {
        display: trimmed,
        url: platformUrl,
      }
    }
  }

  const fallbackUrl = extractUrl(trimmed)
  if (fallbackUrl) {
    return {
      display: trimmed,
      url: fallbackUrl,
    }
  }

  return { display: trimmed }
}

function extractUrl(value: string): string | undefined {
  const directMatch = value.match(/https?:\/\/\S+/i)
  if (directMatch) {
    return normalizeUrl(directMatch[0])
  }

  const platformHandleUrl = buildPlatformUrl("", value)
  if (platformHandleUrl) {
    return platformHandleUrl
  }

  const domainMatch = value.match(/(?:www\.)?[\w.-]+\.[a-z]{2,}(?:\/\S*)?/i)
  if (domainMatch) {
    return normalizeUrl(domainMatch[0])
  }

  return undefined
}

function normalizeUrl(url: string): string {
  const cleanUrl = url.trim().replace(/[),.;]+$/, "")
  if (/^https?:\/\//i.test(cleanUrl)) {
    return cleanUrl
  }

  return `https://${cleanUrl}`
}

function extractUrlMatches(text: string): TextUrlMatch[] {
  const matches: TextUrlMatch[] = []
  const regex = new RegExp(URL_REGEX.source, "gi")
  let match = regex.exec(text)

  while (match) {
    const raw = match[0]
    const start = match.index
    const end = start + raw.length
    matches.push({
      text: raw,
      start,
      end,
      url: normalizeUrl(raw),
    })
    match = regex.exec(text)
  }

  return matches
}

function LinkifiedText({ text }: { text: string }) {
  const matches = extractUrlMatches(text)
  if (matches.length === 0) {
    return <>{text}</>
  }

  const parts: Array<{ type: "text" | "link"; value: string; href?: string }> = []
  let cursor = 0

  for (const match of matches) {
    if (match.start > cursor) {
      parts.push({ type: "text", value: text.slice(cursor, match.start) })
    }
    parts.push({ type: "link", value: match.text, href: match.url })
    cursor = match.end
  }

  if (cursor < text.length) {
    parts.push({ type: "text", value: text.slice(cursor) })
  }

  return (
    <>
      {parts.map((part, index) =>
        part.type === "link" ? (
          <a
            key={`${part.value}-${index}`}
            href={part.href}
            target="_blank"
            rel="noreferrer"
            className="underline decoration-slate-500/50 underline-offset-2 hover:text-slate-900"
          >
            {part.value}
          </a>
        ) : (
          <Fragment key={`${part.value}-${index}`}>{part.value}</Fragment>
        )
      )}
    </>
  )
}

function buildPlatformUrl(platform: string, value: string): string | undefined {
  const normalizedPlatform = platform.trim().toLowerCase()
  const normalizedValue = value.trim()
  if (!normalizedValue) {
    return undefined
  }

  if (/linkedin/.test(normalizedPlatform)) {
    if (/linkedin\.com/i.test(normalizedValue)) {
      return normalizeUrl(normalizedValue)
    }

    const handleMatch = normalizedValue.match(/^@?([a-z0-9-]{3,})$/i)
    if (handleMatch) {
      return `https://www.linkedin.com/in/${handleMatch[1]}`
    }

    return `https://www.linkedin.com/search/results/all/?keywords=${encodeURIComponent(normalizedValue)}`
  }

  if (/youtube/.test(normalizedPlatform) || /youtube\.com|youtu\.be/i.test(normalizedValue)) {
    if (/youtube\.com|youtu\.be/i.test(normalizedValue)) {
      return normalizeUrl(normalizedValue)
    }

    const handleMatch = normalizedValue.match(/^@([a-z0-9._-]+)$/i)
    if (handleMatch) {
      return `https://www.youtube.com/@${handleMatch[1]}`
    }
  }

  if (/github/.test(normalizedPlatform) || /github\.com/i.test(normalizedValue)) {
    if (/github\.com/i.test(normalizedValue)) {
      return normalizeUrl(normalizedValue.replace(/^@/, ""))
    }

    const handleMatch = normalizedValue.match(/^@?([a-z0-9-]{1,39})$/i)
    if (handleMatch) {
      return `https://github.com/${handleMatch[1]}`
    }
  }

  if (/portfolio|website|site|web/.test(normalizedPlatform)) {
    return normalizeUrl(normalizedValue)
  }

  return undefined
}

function parseSectionBulletItems(sectionHeading: string, content: string): BulletItem[] {
  const normalizedHeading = sectionHeading.trim().toLowerCase()
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)

  if (normalizedHeading === "skills") {
    return parseSkillsBulletItems(lines)
  }

  if (["education", "languages", "language"].includes(normalizedHeading)) {
    return lines
      .map((line) => line.replace(/^[•\-*]+\s*/, "").trim())
      .filter(Boolean)
      .map((text) => ({ level: "primary" as const, text }))
  }

  const items: BulletItem[] = []
  let lastLevel: BulletItem["level"] = "primary"

  for (let index = 0; index < lines.length; index += 1) {
    const cleanedLine = lines[index].replace(/^[•\-*]+\s*/, "").trim()
    if (!cleanedLine) {
      continue
    }

    const level: BulletItem["level"] = isPrimaryBulletLine(cleanedLine, sectionHeading, index, lastLevel)
      ? "primary"
      : "secondary"

    items.push({ level, text: cleanedLine })
    lastLevel = level
  }

  return items
}

function splitDateSuffix(text: string): { leftText: string; dateText?: string } {
  const trimmed = text.trim()
  if (!trimmed) {
    return { leftText: text }
  }

  const pipeIndex = trimmed.lastIndexOf("|")
  if (pipeIndex > -1) {
    const left = trimmed.slice(0, pipeIndex).trim()
    const right = trimmed.slice(pipeIndex + 1).trim()
    if (isLikelyDateRange(right) && left) {
      return { leftText: left, dateText: right }
    }
  }

  const trailingDateRangeMatch = trimmed.match(
    /((?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4}|(?:19|20)\d{2})\s*[–—-]\s*(?:Present|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4}|(?:19|20)\d{2})))$/i
  )

  if (trailingDateRangeMatch) {
    const dateText = trailingDateRangeMatch[1].trim()
    const leftText = trimmed
      .slice(0, trailingDateRangeMatch.index)
      .replace(/[|–—-]\s*$/, "")
      .trim()

    if (leftText && isLikelyDateRange(dateText)) {
      return { leftText, dateText }
    }
  }

  return { leftText: trimmed }
}

function shouldBoldPrimaryForSection(sectionHeading: string): boolean {
  const normalizedHeading = sectionHeading.trim().toLowerCase()
  return normalizedHeading === "work experience" || normalizedHeading === "projects & research"
}

function isLikelyDateRange(value: string): boolean {
  const normalized = value.trim().toLowerCase()
  const hasMonth = /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b/.test(normalized)
  const hasYear = /\b(?:19|20)\d{2}\b/.test(normalized)
  const hasPresent = /\bpresent\b/.test(normalized)
  const hasRangeSeparator = /[-–—]/.test(normalized)
  return (hasMonth || hasYear || hasPresent) && (hasRangeSeparator || hasPresent)
}

function parseSkillsBulletItems(lines: string[]): BulletItem[] {
  const cleanedLines = lines
    .map((line) => line.replace(/^[•\-*]+\s*/, "").trim())
    .filter(Boolean)

  if (cleanedLines.some((line) => line.includes(":"))) {
    return cleanedLines.map((text) => {
      const separatorIndex = text.indexOf(":")
      if (separatorIndex === -1) {
        return { level: "primary" as const, text }
      }

      const label = text.slice(0, separatorIndex).trim()
      const detail = text.slice(separatorIndex + 1).trim()
      if (!label || !detail) {
        return { level: "primary" as const, text }
      }

      return {
        level: "primary" as const,
        text,
        label,
        detail,
      }
    })
  }

  const tokens = cleanedLines
    .flatMap((line) => line.split(","))
    .map((token) => token.trim())
    .filter(Boolean)

  const groups: string[] = []
  let currentGroup: string[] = []
  let currentLength = 0

  for (const token of tokens) {
    const tokenLength = token.length
    const separatorLength = currentGroup.length > 0 ? 2 : 0
    const exceedsLineLength = currentLength + separatorLength + tokenLength > 72
    const exceedsItemCount = currentGroup.length >= 4

    if ((exceedsLineLength || exceedsItemCount) && currentGroup.length > 0) {
      groups.push(currentGroup.join(", "))
      currentGroup = [token]
      currentLength = tokenLength
      continue
    }

    currentGroup.push(token)
    currentLength += separatorLength + tokenLength
  }

  if (currentGroup.length > 0) {
    groups.push(currentGroup.join(", "))
  }

  return groups.map((text) => ({ level: "primary" as const, text }))
}

function isPrimaryBulletLine(
  line: string,
  sectionHeading: string,
  index: number,
  lastLevel: BulletItem["level"]
): boolean {
  if (index === 0) {
    return true
  }

  const normalizedHeading = sectionHeading.trim().toLowerCase()
  const hasDatePattern =
    /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b/i.test(line) ||
    /\b(?:19|20)\d{2}\b/.test(line) ||
    /\bpresent\b/i.test(line)
  const hasRoleSeparator = /\|/.test(line) || /[–—-]\s*(?:present|(?:19|20)\d{2})/i.test(line)
  const skillCategoryLine = normalizedHeading === "skills" && line.includes(":")

  if (skillCategoryLine || hasDatePattern || hasRoleSeparator) {
    return true
  }

  if (lastLevel === "secondary") {
    return false
  }

  return false
}

function wrapText(
  text: string,
  font: { widthOfTextAtSize: (text: string, size: number) => number },
  fontSize: number,
  maxWidth: number
): string[] {
  const paragraphs = text.split("\n")
  const allLines: string[] = []

  for (const paragraph of paragraphs) {
    if (paragraph.trim() === "") {
      allLines.push("")
      continue
    }

    const words = paragraph.split(" ")
    let currentLine = ""

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word
      const width = font.widthOfTextAtSize(testLine, fontSize)

      if (width > maxWidth && currentLine) {
        allLines.push(currentLine)
        currentLine = word
      } else {
        currentLine = testLine
      }
    }
    if (currentLine) {
      allLines.push(currentLine)
    }
  }

  return allLines
}
