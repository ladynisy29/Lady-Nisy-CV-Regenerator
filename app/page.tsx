"use client"

import { useState, useCallback } from "react"
import { FileUpload } from "@/components/file-upload"
import { StepIndicator } from "@/components/step-indicator"
import { CVPreview } from "@/components/cv-preview"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ArrowLeft, ArrowRight, FileText, Sparkles, Loader2 } from "lucide-react"

const STEPS = [
  { label: "Upload CV", description: "Your existing resume" },
  { label: "Job Offer", description: "Paste the listing" },
  { label: "Generate", description: "AI-tailored CV" },
]

interface CVData {
  fullName: string
  title: string
  email: string
  phone: string
  location: string
  summary: string
  sections: { heading: string; content: string }[]
}

interface ParsedCVData {
  otherSections?: { heading: string; content: string }[]
}

interface GenerateCvResponse {
  parsedCv: ParsedCVData
  tailoredCv: CVData
}

function mergeOnlineProfiles(parsedCv: ParsedCVData, tailoredCv: CVData): CVData {
  const hasOnlineProfiles = tailoredCv.sections.some((section) =>
    isOnlineProfilesHeading(section.heading)
  )

  if (hasOnlineProfiles) {
    return tailoredCv
  }

  const onlineProfilesSection = parsedCv.otherSections?.find((section) =>
    isOnlineProfilesHeading(section.heading)
  )

  if (!onlineProfilesSection) {
    return tailoredCv
  }

  return {
    ...tailoredCv,
    sections: [...tailoredCv.sections, onlineProfilesSection],
  }
}

function isOnlineProfilesHeading(heading: string): boolean {
  const normalizedHeading = heading.trim().toLowerCase()
  return (
    normalizedHeading === "online profiles" ||
    normalizedHeading === "online profile" ||
    normalizedHeading === "profiles" ||
    normalizedHeading === "links"
  )
}

export default function Page() {
  const [step, setStep] = useState(1)
  const [cvFile, setCvFile] = useState<File | null>(null)
  const [jobOffer, setJobOffer] = useState("")
  const [cvData, setCvData] = useState<CVData | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [streamingText, setStreamingText] = useState("")
  const [error, setError] = useState("")

  const extractTextFromPdf = useCallback(async (file: File): Promise<string> => {
    const arrayBuffer = await file.arrayBuffer()
    const response = await fetch("/api/parse-pdf", {
      method: "POST",
      body: arrayBuffer,
      headers: { "Content-Type": "application/octet-stream" },
    })
    if (!response.ok) throw new Error("Failed to parse PDF")
    const data = await response.json()
    return data.text
  }, [])

  const handleGenerate = useCallback(async () => {
    if (!cvFile || !jobOffer.trim()) return

    setIsGenerating(true)
    setStreamingText("")
    setCvData(null)
    setError("")

    try {
      const cvText = await extractTextFromPdf(cvFile)

      const formData = new FormData()
      formData.append("cvText", cvText)
      formData.append("jobOffer", jobOffer)

      const response = await fetch("/api/generate-cv", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || "Failed to generate CV")
      }

      const parsed = (await response.json()) as GenerateCvResponse
      setCvData(mergeOnlineProfiles(parsed.parsedCv, parsed.tailoredCv))
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.")
    } finally {
      setIsGenerating(false)
      setStreamingText("")
    }
  }, [cvFile, jobOffer, extractTextFromPdf])

  const canProceed = step === 1 ? !!cvFile : step === 2 ? jobOffer.trim().length > 20 : false

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="animate-fade-in border-b border-primary/30 bg-card/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4 sm:px-6">
          <div className="animate-fade-up flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent shadow-[0_0_20px_hsl(var(--primary)/0.45)]">
              <FileText className="h-5 w-5 text-black" />
            </div>
            <span className="text-lg font-bold text-primary">Lady Nisy&apos;s CV Regenerator</span>
          </div>
          <div className="animate-fade-up animate-delay-1">
            <StepIndicator currentStep={step} steps={STEPS} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        {step === 1 && (
          <div className="animate-fade-up mx-auto max-w-2xl rounded-2xl border border-primary/25 bg-card/70 p-6 shadow-[0_24px_80px_hsl(var(--background)/0.8)] sm:p-8">
            <div className="mb-8 flex flex-col gap-2">
              <h1 className="text-balance text-2xl font-bold text-primary sm:text-3xl">
                Upload Your Existing CV
              </h1>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Start by uploading your current resume. We will extract your experience, skills,
                and education to craft a tailored version for your target role.
              </p>
            </div>

            <FileUpload
              label="Your CV"
              description="Upload your current resume in PDF format"
              accept=".pdf"
              file={cvFile}
              onFileChange={setCvFile}
            />

            <div className="mt-8 flex justify-end">
              <Button
                onClick={() => setStep(2)}
                disabled={!canProceed}
                className="bg-primary text-primary-foreground hover:bg-primary/90"
              >
                Continue
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="animate-fade-up mx-auto max-w-2xl rounded-2xl border border-primary/25 bg-card/70 p-6 shadow-[0_24px_80px_hsl(var(--background)/0.8)] sm:p-8">
            <div className="mb-8 flex flex-col gap-2">
              <h1 className="text-balance text-2xl font-bold text-primary sm:text-3xl">
                Paste the Job Offer
              </h1>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Copy and paste the full job listing below. The more details you include, the
                better we can tailor your CV to match the requirements.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <label
                htmlFor="job-offer"
                className="text-sm font-semibold uppercase tracking-wider text-foreground"
              >
                Job Description
              </label>
              <Textarea
                id="job-offer"
                value={jobOffer}
                onChange={(e) => setJobOffer(e.target.value)}
                placeholder="Paste the full job offer here... Include the job title, requirements, responsibilities, and any other details."
                className="min-h-[280px] resize-y border border-primary/30 bg-background/80 text-sm leading-relaxed shadow-inner"
              />
              <span className="text-xs text-muted-foreground">
                {jobOffer.length > 0
                  ? `${jobOffer.length} characters`
                  : "Minimum 20 characters required"}
              </span>
            </div>

            <div className="mt-8 flex justify-between">
              <Button
                variant="outline"
                onClick={() => setStep(1)}
                className="border-2"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back
              </Button>
              <Button
                onClick={() => {
                  setStep(3)
                  handleGenerate()
                }}
                disabled={!canProceed}
                className="bg-accent text-accent-foreground hover:bg-accent/90"
              >
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Tailored CV
              </Button>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="animate-fade-up flex flex-col gap-6 rounded-2xl border border-primary/25 bg-card/70 p-6 shadow-[0_24px_80px_hsl(var(--background)/0.8)] sm:p-8">
            <div className="flex flex-col gap-2">
              <h1 className="text-balance text-2xl font-bold text-primary sm:text-3xl">
                Your AI-Tailored CV
              </h1>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Review your new resume below. When you are satisfied, download it as a PDF.
              </p>
            </div>

            {error && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
                <p className="text-sm text-destructive">{error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setError("")
                    handleGenerate()
                  }}
                  className="mt-2 border-2"
                >
                  Try Again
                </Button>
              </div>
            )}

            <CVPreview
              data={cvData}
              isGenerating={isGenerating}
              streamingText={streamingText}
            />

            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                onClick={() => {
                  setStep(2)
                  setCvData(null)
                  setStreamingText("")
                  setError("")
                }}
                className="border-2"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Edit Inputs
              </Button>
              {cvData && (
                <Button
                  onClick={() => {
                    setCvData(null)
                    setStreamingText("")
                    setError("")
                    handleGenerate()
                  }}
                  disabled={isGenerating}
                  variant="outline"
                  className="border-2"
                >
                  {isGenerating ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Sparkles className="mr-2 h-4 w-4" />
                  )}
                  Regenerate
                </Button>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="animate-fade-in animate-delay-2 border-t border-primary/30 bg-card/75 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-center px-4 py-4 sm:px-6">
          <p className="text-xs text-muted-foreground">
            Your data is processed securely and never stored on our servers.
          </p>
        </div>
      </footer>
    </div>
  )
}
