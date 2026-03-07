"use client"

import { Download, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useState } from "react"

interface CVSection {
  heading: string
  content: string
}

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

  const handleDownload = async () => {
    if (!data) return
    setIsDownloading(true)
    try {
      const { PDFDocument, rgb, StandardFonts } = await import("pdf-lib")

      const pdfDoc = await PDFDocument.create()
      const helvetica = await pdfDoc.embedFont(StandardFonts.Helvetica)
      const helveticaBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold)

      const pageWidth = 595.28
      const pageHeight = 841.89
      const margin = 50
      const contentWidth = pageWidth - margin * 2

      let page = pdfDoc.addPage([pageWidth, pageHeight])
      let y = pageHeight - margin

      const darkBlue = rgb(0.169, 0.176, 0.329)
      const coral = rgb(1, 0.463, 0.361)
      const gray = rgb(0.4, 0.4, 0.45)

      const addNewPageIfNeeded = (requiredSpace: number) => {
        if (y - requiredSpace < margin) {
          page = pdfDoc.addPage([pageWidth, pageHeight])
          y = pageHeight - margin
        }
      }

      // Name
      page.drawText(data.fullName, {
        x: margin,
        y,
        size: 24,
        font: helveticaBold,
        color: darkBlue,
      })
      y -= 28

      // Title
      if (data.title) {
        page.drawText(data.title, {
          x: margin,
          y,
          size: 12,
          font: helvetica,
          color: coral,
        })
        y -= 20
      }

      // Contact info
      const contactParts = [data.email, data.phone, data.location].filter(Boolean)
      if (contactParts.length > 0) {
        page.drawText(contactParts.join("  |  "), {
          x: margin,
          y,
          size: 9,
          font: helvetica,
          color: gray,
        })
        y -= 20
      }

      // Divider
      page.drawLine({
        start: { x: margin, y },
        end: { x: pageWidth - margin, y },
        thickness: 1,
        color: coral,
      })
      y -= 20

      // Summary
      if (data.summary) {
        const summaryLines = wrapText(data.summary, helvetica, 10, contentWidth)
        for (const line of summaryLines) {
          addNewPageIfNeeded(14)
          page.drawText(line, {
            x: margin,
            y,
            size: 10,
            font: helvetica,
            color: gray,
          })
          y -= 14
        }
        y -= 10
      }

      // Sections
      for (const section of data.sections) {
        addNewPageIfNeeded(40)

        page.drawText(section.heading.toUpperCase(), {
          x: margin,
          y,
          size: 11,
          font: helveticaBold,
          color: darkBlue,
        })
        y -= 6

        page.drawLine({
          start: { x: margin, y },
          end: { x: margin + 60, y },
          thickness: 2,
          color: coral,
        })
        y -= 14

        const lines = wrapText(section.content, helvetica, 10, contentWidth)
        for (const line of lines) {
          addNewPageIfNeeded(14)
          page.drawText(line, {
            x: margin,
            y,
            size: 10,
            font: helvetica,
            color: gray,
          })
          y -= 14
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
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold uppercase tracking-wider text-foreground">
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

      <div className="overflow-hidden rounded-lg border border-primary/30 bg-card/85 shadow-[0_16px_48px_hsl(var(--background)/0.7)]">
        {/* Header */}
        <div className="border-b border-primary/25 bg-gradient-to-r from-primary/15 to-accent/10 p-6">
          <h2 className="text-2xl font-bold text-primary">{data.fullName}</h2>
          {data.title && (
            <p className="mt-1 text-sm font-medium text-foreground/90">{data.title}</p>
          )}
          <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted-foreground">
            {data.email && <span>{data.email}</span>}
            {data.phone && (
              <>
                <span className="text-border">|</span>
                <span>{data.phone}</span>
              </>
            )}
            {data.location && (
              <>
                <span className="text-border">|</span>
                <span>{data.location}</span>
              </>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="flex flex-col gap-6 p-6">
          {data.summary && (
            <p className="text-sm leading-relaxed text-muted-foreground">{data.summary}</p>
          )}

          {data.sections.map((section, index) => (
            <div key={index} className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold uppercase tracking-widest text-primary">
                  {section.heading}
                </h3>
                <div className="h-px flex-1 bg-border" />
              </div>
              <div className="whitespace-pre-line text-sm leading-relaxed text-muted-foreground">
                {section.content}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
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
