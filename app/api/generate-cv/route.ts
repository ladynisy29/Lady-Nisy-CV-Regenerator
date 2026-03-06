import { generateObject } from "ai"
import { createOpenAI } from "@ai-sdk/openai"
import { z } from "zod"

const DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
const DEFAULT_GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
const DEFAULT_GITHUB_MODELS_MODEL = "gpt-4.1-mini"

function normalizeGithubModel(modelId: string) {
  const trimmed = modelId.trim()
  // GitHub Models OpenAI-compatible endpoint usually expects bare model IDs.
  return trimmed.includes("/") ? trimmed.split("/").pop() || trimmed : trimmed
}

function resolveModel() {
  const provider = (process.env.AI_PROVIDER || "openai").toLowerCase()

  if (provider === "github-models") {
    const githubApiKey = process.env.GITHUB_MODELS_API_KEY || process.env.GITHUB_TOKEN
    if (!githubApiKey) {
      throw new Error(
        "GITHUB_MODELS_API_KEY (or GITHUB_TOKEN) is required when AI_PROVIDER=github-models."
      )
    }
    if (githubApiKey.includes("your_actual_api_key_here")) {
      throw new Error(
        "GITHUB_MODELS_API_KEY is still set to a placeholder value. Update .env.local with a real key."
      )
    }

    const githubProvider = createOpenAI({
      name: "github-models",
      apiKey: githubApiKey,
      baseURL: process.env.GITHUB_MODELS_BASE_URL || DEFAULT_GITHUB_MODELS_BASE_URL,
      headers: {
        ...(process.env.GITHUB_MODELS_APP_NAME
          ? { "X-Title": process.env.GITHUB_MODELS_APP_NAME }
          : {}),
      },
    })

    const configuredModel =
      process.env.GITHUB_MODELS_MODEL || process.env.OPENAI_MODEL || DEFAULT_GITHUB_MODELS_MODEL
    const githubModel = normalizeGithubModel(configuredModel)
    return githubProvider.chat(githubModel)
  }

  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not configured on the server.")
  }
  if (process.env.OPENAI_API_KEY.includes("your_actual_api_key_here")) {
    throw new Error(
      "OPENAI_API_KEY is still set to a placeholder value. Update .env.local with a real key."
    )
  }

  const openAIProvider = createOpenAI({ apiKey: process.env.OPENAI_API_KEY })
  const openAIModel = process.env.OPENAI_MODEL || DEFAULT_OPENAI_MODEL
  return openAIProvider.chat(openAIModel)
}

const ParsedCVSchema = z.object({
  fullName: z.string().describe("Candidate full name"),
  title: z.string().describe("Current or most relevant professional title"),
  email: z.string().describe("Email address"),
  phone: z.string().describe("Phone number"),
  location: z.string().describe("City/region or location"),
  summary: z.string().describe("Short profile summary extracted from original CV"),
  skills: z.array(z.string()).describe("List of skills explicitly present in CV"),
  experience: z
    .array(
      z.object({
        role: z.string(),
        company: z.string(),
        period: z.string(),
        achievements: z.array(z.string()),
      })
    )
    .describe("Professional experience entries from CV"),
  education: z
    .array(
      z.object({
        institution: z.string(),
        qualification: z.string(),
        period: z.string(),
      })
    )
    .describe("Education entries from CV"),
  otherSections: z
    .array(
      z.object({
        heading: z.string(),
        content: z.string(),
      })
    )
    .describe("Any other sections found in CV"),
})

const TailoredCVSchema = z.object({
  fullName: z.string().describe("The full name of the candidate"),
  title: z.string().describe("A professional title tailored to the job offer"),
  email: z.string().describe("Email address from the original CV"),
  phone: z.string().describe("Phone number from the original CV"),
  location: z.string().describe("Location from the original CV"),
  summary: z
    .string()
    .describe(
      "A compelling professional summary tailored to the job offer, 2-4 sentences"
    ),
  sections: z
    .array(
      z.object({
        heading: z
          .string()
          .describe("Section heading like Experience, Education, Skills, etc."),
        content: z
          .string()
          .describe("Section content with details tailored to the job offer"),
      })
    )
    .describe(
      "CV sections in order: Experience, Education, Skills, and any other relevant sections"
    ),
})

export async function POST(req: Request) {
  const formData = await req.formData()
  const cvText = formData.get("cvText") as string
  const jobOffer = formData.get("jobOffer") as string

  if (!cvText || !jobOffer) {
    return Response.json(
      { error: "Both CV text and job offer are required." },
      { status: 400 }
    )
  }

  try {
    const model = resolveModel()

    // Step 1: Convert raw CV text to a structured JSON representation.
    const parsedCvResult = await generateObject({
      model,
      schema: ParsedCVSchema,
      system: `You are an expert CV parser.

Your task is to convert unstructured CV text into a structured JSON format.

Rules:
- Preserve facts exactly as written in the CV.
- Do not invent or infer missing facts.
- If a field is unavailable, use an empty string or empty array.
- Keep achievements as concise bullet-like statements in the achievements arrays.
- Keep role/company/period separate when possible.`,
      prompt: `Convert the following CV text into the required JSON structure:

---
${cvText}
---`,
    })

    const parsedCv = parsedCvResult.object

    // Step 2: Compare parsed CV JSON with job description and generate tailored CV.
    const tailoredCvResult = await generateObject({
      model,
      schema: TailoredCVSchema,
      system: `You are an expert CV/resume writer. Your task is to create a professionally tailored CV based on the candidate's structured CV JSON and a specific job offer.

Guidelines:
- Preserve all factual information from the original CV JSON (name, contact, dates, companies, education).
- Rewrite the professional summary to directly address the job requirements.
- Reorganize and rephrase achievements to highlight relevant skills and measurable impact.
- Use strong action verbs and quantified achievements where possible.
- Ensure skills section emphasizes technologies/competencies mentioned in the job offer.
- Keep the tone professional and concise.
- Do NOT fabricate experience or skills not present in the original CV.
- If information is missing from the original CV JSON, use an empty string rather than making things up.`,
      prompt: `Here is the candidate's structured CV JSON:

---
${JSON.stringify(parsedCv, null, 2)}
---

Here is the job offer they are applying for:

---
${jobOffer}
---

Create a tailored CV that highlights the candidate's most relevant qualifications for this specific role.`,
    })

    return Response.json({
      parsedCv,
      tailoredCv: tailoredCvResult.object,
    })
  } catch (error) {
    console.error("CV generation error:", error)
    return Response.json(
      {
        error:
          "Failed to generate CV: " +
          (error instanceof Error ? error.message : "Unknown error"),
      },
      { status: 500 }
    )
  }
}
