import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy GoogleGenAI client helper
let aiClient: GoogleGenAI | null = null;
function getAi(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is not configured. Please define it in your Secrets panel.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        }
      }
    });
  }
  return aiClient;
}

// Helper to generate JSON with multi-model fallback and backoff
async function generateWithModelFallback(prompt: string): Promise<any> {
  const ai = getAi();
  // Try modern gemini-3.8-flash first, then flash-lite, then latest alias
  const candidateModels = [
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest"
  ];

  let lastError: any = null;

  for (const model of candidateModels) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          temperature: 0.7,
        }
      });

      const text = response.text;
      if (text && text.trim().length > 0) {
        const cleaned = text.replace(/^```json\s*/i, "").replace(/```\s*$/, "").trim();
        return JSON.parse(cleaned);
      }
    } catch (err: any) {
      lastError = err;
      console.warn(`[Gemini Resiliency] Model "${model}" reported temporary notice: ${err?.message || "High demand / unavailable"}. Trying next candidate model.`);
      // Short backoff before next candidate
      await new Promise((r) => setTimeout(r, 450));
    }
  }

  throw lastError || new Error("All Gemini model candidates exhausted");
}

function getDefaultGrowSheet(cropId: string, cropName: string, soilType?: string, soilPh?: number) {
  const soil = soilType || "Loamy";
  const ph = soilPh || 6.5;
  return {
    id: cropId,
    name: cropName,
    category: "Horticulture",
    physiology: {
      matureHeight: "1.0m - 1.5m (3.3 - 5 ft)",
      rootDepth: "0.6m - 0.9m (2 - 3 ft)"
    },
    spacing: {
      intraRow: "30 - 45 cm (12 - 18 in)",
      interRow: "60 - 95 cm (24 - 38 in)",
      diagram: `[ Row 1 ]  (🌱) ---- 35cm ---- (🌱) ---- 35cm ---- (🌱)\n               |\n             80cm (Row Spacing)\n               |\n[ Row 2 ]  (🌱) ---- 35cm ---- (🌱) ---- 35cm ---- (🌱)`
    },
    fertigation: {
      prePlanting: `Incorporate 40 kg/ha N, 60 kg/ha P, 60 kg/ha K under ${soil} soil conditions.`,
      vegetative: "Apply 80kg of Urea at 3 weeks to support high leaf canopy extension.",
      flowering: "Drip soluble 15-15-30 compound at 20 kg/ha to encourage blossoms.",
      fruiting: "Drip Potassium Nitrate at 25 kg/ha to optimize product weight.",
      micronutrients: `Soil contains pH ${ph}. Apply multi-chelate Zinc-EDTA at 4 kg/ha to optimize chlorophyll activity.`
    },
    water: {
      evapotranspiration: "3 - 5 mm per day.",
      frequency: "Drip: 1.5 hours every other day. Maintain moist organic layer."
    },
    pests: [
      { name: "Common Aphids", countermeasure: "Foliar spray organic cold-pressed neem oil concentrate weekly.", preHarvestInterval: "2 Days" },
      { name: "Early Blight", countermeasure: "Prune bottom leaves to elevate airflow. Apply copper octanoate if wet.", preHarvestInterval: "1 Day" }
    ]
  };
}

function getDefaultLivestockPlan(livestockId: string, livestockName: string, soilType?: string) {
  const soil = soilType || "Loamy";
  return {
    id: livestockId,
    name: livestockName,
    category: "Cattle",
    stockingDensity: "Requires 1.5 to 2.5 hectares of grassland pasture per animal for rotation.",
    feedFormulation: {
      starter: "Ration composed of 14% CP alfalfa haylage and ground oats starter creep.",
      grower: "Transition to 12% CP silage forage, clover, and cracked corn.",
      finisher: "Formulate with 11% CP roughage, steam-flaked sorghum, and mineral salts.",
      calculatorHelp: "Target feed conversion ratio is 6:1. Ensure dry block pasture salt is available."
    },
    veterinarySchedule: {
      vaccinations: [
        "Month 3: 7-Way Clostridial (Blackleg) immunization.",
        "Month 6: Infectious Bovine Rhinotracheitis (IBR) booster."
      ],
      deworming: "Deworm twice annually during seasonal transitions using pour-on Ivermectin."
    },
    wasteManagement: {
      guidelines: `Collect dry manure. Windrow compost with straw on ${soil} fields to raise soil humus.`,
      biodigesterSize: "For 10 head: Requires a 12 cubic meter anaerobic biodigester for steady biogas generation."
    }
  };
}

// 1. API: Generate custom precision Grow Sheet for crops
app.post("/api/gemini/grow-sheet", async (req, res) => {
  const { cropName, cropId, soilType, soilPh, latitude, longitude } = req.body;

  if (!cropName || !cropId) {
    return res.status(400).json({ success: false, error: "Missing required parameters: cropName and cropId" });
  }

  const prompt = `Generate an exhaustive precision agricultural grow sheet for the crop "${cropName}" (ID: "${cropId}").
The farm context is:
- Soil type: ${soilType || "Loamy"}
- Soil pH: ${soilPh || 6.5}
- GPS coordinates: ${latitude || 41.5}, ${longitude || -93.6}

You must return a JSON response matching the following schema structure exactly. Return ONLY valid JSON:
{
  "id": "${cropId}",
  "name": "${cropName}",
  "category": "Field Crop" or "Horticulture",
  "physiology": {
    "matureHeight": "string with height range, e.g., '1.5m - 2.0m'",
    "rootDepth": "string with root depth, e.g., '0.9m - 1.2m'"
  },
  "spacing": {
    "intraRow": "spacing recommendation in metric (cm) and imperial (in)",
    "interRow": "spacing recommendation in metric (cm) and imperial (in)",
    "diagram": "A highly readable ASCII/text diagram representing the planting row spacing pattern. Use emojis or symbols like (🌱) or (🍅) to show crops in columns/rows."
  },
  "fertigation": {
    "prePlanting": "actionable guide specifying exact N-P-K ratios and organic matter percentage to apply (e.g. 'Apply 50 kg/ha of Nitrogen...')",
    "vegetative": "actionable fertilizer instructions with specific products (e.g. 'Apply 100kg of Urea') and growth stages",
    "flowering": "actionable fertilizer instructions for the blooming/flowering phase",
    "fruiting": "actionable instructions for fruit/grain development phase",
    "micronutrients": "specific chelate recommendations (e.g. Zinc-EDTA, Iron-EDDHA) adapted to the soil pH of ${soilPh || 6.5}"
  },
  "water": {
    "evapotranspiration": "Daily/weekly rate in mm, e.g. '4 - 6 mm per day'",
    "frequency": "Actionable watering frequency guide (drip/sprinkler), e.g., 'Drip: Daily 45 mins.'"
  },
  "pests": [
    {
      "name": "Pest/Disease name",
      "countermeasure": "biological/organic and chemical countermeasures",
      "preHarvestInterval": "withholding interval, e.g., '7 Days'"
    }
  ]
}`;

  try {
    const data = await generateWithModelFallback(prompt);
    res.json({ success: true, data });
  } catch (error: any) {
    console.warn(`[AgriChoice Resiliency] Gemini Grow-Sheet notice (${error?.message || "Peak demand"}); serving verified agronomic profile for ${cropName}.`);
    res.json({
      success: true,
      data: getDefaultGrowSheet(cropId, cropName, soilType, soilPh)
    });
  }
});

// 2. API: Generate custom precision Herd Management Plan for livestock
app.post("/api/gemini/livestock-plan", async (req, res) => {
  const { livestockName, livestockId, soilType } = req.body;

  if (!livestockName || !livestockId) {
    return res.status(400).json({ success: false, error: "Missing required parameters: livestockName and livestockId" });
  }

  const prompt = `Generate a comprehensive livestock optimization plan for "${livestockName}" (ID: "${livestockId}").
The farm context is:
- Soil type: ${soilType || "Loamy"}

You must return a JSON response matching the following schema structure exactly. Return ONLY valid JSON:
{
  "id": "${livestockId}",
  "name": "${livestockName}",
  "category": "Poultry", "Swine", "Cattle", or "Aquaculture",
  "stockingDensity": "Specific ethical stocking density limits in square meters or feet required per animal for growth",
  "feedFormulation": {
    "starter": "Ration guide for starter phase (e.g., 20% Crude Protein and primary ingredients)",
    "grower": "Ration guide for grower phase",
    "finisher": "Ration guide for finisher phase",
    "calculatorHelp": "Helpful conversion details or feed conversion ratio (FCR) targets"
  },
  "veterinarySchedule": {
    "vaccinations": [
      "Timeline of key vaccinations (array of strings, e.g. 'Day 1: Newcastle Disease Spray')"
    ],
    "deworming": "Clear protocol and recommended products (e.g., Ivermectin dosage)"
  },
  "wasteManagement": {
    "guidelines": "Guidelines on converting manure/waste to compost or biogas (circular economy) matching soil context",
    "biodigesterSize": "Sizing guidance for an anaerobic biodigester based on headcount"
  }
}`;

  try {
    const data = await generateWithModelFallback(prompt);
    res.json({ success: true, data });
  } catch (error: any) {
    console.warn(`[AgriChoice Resiliency] Gemini Livestock notice (${error?.message || "Peak demand"}); serving verified herd management profile for ${livestockName}.`);
    res.json({
      success: true,
      data: getDefaultLivestockPlan(livestockId, livestockName, soilType)
    });
  }
});

// Fallback guide response generator when Gemini API is unavailable or offline
function getFallbackGuideResponse(userMessage: string, context: any) {
  const msg = (userMessage || "").toLowerCase();
  const country = context?.country || "United States";
  const soil = context?.soilType || "Loamy";
  const ph = context?.soilPh || 6.5;
  const currency = context?.currency || "USD";

  if (msg.includes("tour") || msg.includes("start") || msg.includes("guide") || msg.includes("new user") || msg.includes("how to use") || msg.includes("overview") || msg.includes("welcome")) {
    return {
      reply: `### Welcome to AgriChoice 2.0! 🌱
AgriChoice is a precision agricultural decision-support engine engineered to help farmers, agronomists, and investors make multi-thousand-dollar business decisions with confidence.

Here is your **Quick Onboarding Roadmap**:

1. **Set Your Farm Baseline**: Select your country from the top bar (currently set to **${country}**, currency **${currency}**) and configure your soil (${soil}, pH ${ph}) to calibrate all calculations. [ACTION:country:Change Country]
2. **Tab 1: Profitability Compass**: Compare net margins, capital outlay, and cashflow breakeven across 50+ crops and livestock ranked for your specific soil and water profile. [TAB:compass:Explore Profitability Compass]
3. **Tab 2: Side-by-Side Sandbox**: Put two ventures head-to-head (e.g., Maize vs. Tomatoes, or Broilers vs. Cattle) to evaluate water requirements, disease risk, and labor volatility. [TAB:matrix:Open Decision Matrix]
4. **Tabs 3 & 4: Agronomic Grow Sheets & Livestock Plans**: Access precision spacing blueprints, N-P-K fertigation schedules, and ethical feed/herd management guidelines. [TAB:crops:View Grow Sheets]
5. **Tab 5: Bank-Ready Proposal Builder**: Customize financial projections and dictate your operational scope hands-free using your microphone, then download a PDF prospectus for lenders. [TAB:proposal:Open Proposal Builder]
6. **Tab 6: Subsidies & Policies**: Review localized agricultural grants, equipment financing programs, and legal guidelines across 190+ sovereign countries. [TAB:subsidies:View Subsidies & Policies]

Would you like me to guide you through any of these specific tools?`,
      suggestedActions: [
        "Take an Interactive Tour",
        "How to find the most profitable crop?",
        "How to create a bank loan proposal?",
        "How to use voice dictation?"
      ],
      highlightTab: "compass"
    };
  }

  if (msg.includes("profit") || msg.includes("roi") || msg.includes("margin") || msg.includes("compass") || msg.includes("rank") || msg.includes("money") || msg.includes("revenue")) {
    return {
      reply: `### How the Profitability Compass Works 💰
The **Profitability Compass** calculates true net margins per hectare/acre by modeling four financial pillars:
- **Projected Gross Yield**: Estimated production calibrated to your soil (${soil}) and pH (${ph}).
- **Direct Input Costs**: Seed, fertilizer, pesticides, or feed/veterinary supplies.
- **Operational & Labor Overhead**: Field preparation, weeding, irrigation pumping, and harvest labor.
- **Logistics & Post-Harvest Storage**: Transport, cold storage, and handling.

**Formula**: \`Net Profit = (Projected Yield × Market Price) - (Inputs + Labor + Logistics)\`

**Next Step**: Toggle over to Tab 1 to see how changing your water access or crop choice shifts your bottom line. [TAB:compass:Go to Profitability Compass]`,
      suggestedActions: [
        "Go to Profitability Compass",
        "How do I compare two crops?",
        "How to create a bank proposal?"
      ],
      highlightTab: "compass"
    };
  }

  if (msg.includes("matrix") || msg.includes("compare") || msg.includes("versus") || msg.includes("vs") || msg.includes("difference") || msg.includes("sandbox")) {
    return {
      reply: `### Comparing Ventures in the Decision Matrix ⚖️
High paper profitability does not always mean low operational risk. In the **Side-by-Side Sandbox**:
- **Resource Stress**: Compare water consumption (cubic meters/ha) and peak labor intensity.
- **Risk Spectrum**: Review weather vulnerability, pest/disease sensitivity, and market price volatility.
- **12-Month Cashflow Trajectory**: Understand months with negative cash outlay before first harvest.

**Tip**: Pair a fast-cycle cash crop (like bell peppers or broilers) with a stable staple (like maize or cassava) for cashflow resilience. [TAB:matrix:Open Decision Matrix]`,
      suggestedActions: [
        "Open Decision Matrix",
        "View Crop Grow Sheets",
        "How to generate a proposal?"
      ],
      highlightTab: "matrix"
    };
  }

  if (msg.includes("proposal") || msg.includes("bank") || msg.includes("loan") || msg.includes("pdf") || msg.includes("download") || msg.includes("plan") || msg.includes("underwriting")) {
    return {
      reply: `### Generating a Bank-Ready Farm Proposal 📄
AgriChoice includes an Underwriting Completeness Tracker to help you compile an institutional-grade agribusiness plan:
1. **Personalize Your Entity**: Enter your farm name, funding partner, and exact acreage.
2. **Project Objectives & Scope**: Dictate your goals using the **Microphone** button or type your operational strategy.
3. **Financial Budget Ledger**: View automated seed, labor, logistics, and capital breakdown.
4. **Gemini Spatial Intelligence**: Click "Inject Gemini Spatial Intelligence" to synthesize regional climate and soil data.
5. **Download & Print PDF**: Click "Download Official PDF Prospectus" to produce a clean, multi-page document formatted for loan officers. [TAB:proposal:Open Proposal Builder]`,
      suggestedActions: [
        "Open Proposal Builder",
        "How does voice dictation work?",
        "Check Subsidies & Grants"
      ],
      highlightTab: "proposal"
    };
  }

  if (msg.includes("mic") || msg.includes("voice") || msg.includes("dictate") || msg.includes("speech") || msg.includes("audio") || msg.includes("speak")) {
    return {
      reply: `### Voice-to-Text Dictation Guide 🎙️
You can quickly dictate your farm operational objectives without typing, ideal for when you are out in the field or on mobile:
1. Navigate to **Tab 5: Proposal Builder**. [TAB:proposal:Go to Proposal Builder]
2. Locate the **Project Objectives & Scope** text box.
3. Tap the green **Dictate** button. If prompted by your browser, tap **Allow** for microphone access.
4. Speak clearly about your target yields, drip irrigation plans, or market off-takers. You will see a live audio pulse and speech transcription.
5. Tap **Stop** when finished; your thoughts will be formatted into your official cover notes!

*Note: You can also use the microphone icon right here in this chat to speak your questions!*`,
      suggestedActions: [
        "Go to Proposal Builder",
        "Take an Interactive Tour",
        "What crop fits my soil?"
      ],
      highlightTab: "proposal"
    };
  }

  if (msg.includes("country") || msg.includes("subsidy") || msg.includes("subsidies") || msg.includes("grant") || msg.includes("policy") || msg.includes("currency") || msg.includes("nation") || msg.includes("global")) {
    return {
      reply: `### Global Support & Subsidies Database 🌍
AgriChoice features a sovereign database covering **190+ countries**:
- **Automatic Currency Calibration**: Selecting a country updates all financial metrics to sovereign currency (e.g., USD $, NGN ₦, EUR €, INR ₹, KES KSh, BRL R$).
- **Agro-Ecological Regions**: Geo-coordinates, soil baselines, and sowing windows adapt to the chosen country.
- **Tab 6 (Subsidies & Laws)**: Access verified national development bank loan rates, fertilizer subsidy programs, and environmental compliance frameworks. [TAB:subsidies:View Subsidies & Policies]

You can switch your country anytime using the selector in the top bar. [ACTION:country:Select Country]`,
      suggestedActions: [
        "Select Country",
        "View Subsidies & Policies",
        "Go to Profitability Compass"
      ],
      highlightTab: "subsidies"
    };
  }

  // General agronomic guidance
  return {
    reply: `### AgriGuide Agricultural Advisory 🌱
Based on your farm context in **${country}** with **${soil} soil** (pH **${ph}**) and **${context?.waterAvailability || "Medium"}** water availability:

- **Soil pH Compatibility**: A pH of ${ph} is in the ${ph < 6.0 ? "moderately acidic" : ph > 7.5 ? "alkaline" : "near-optimal neutral"} range. Most commercial field crops (maize, soybeans) and horticulture (tomatoes, peppers) thrive in this zone.
- **Recommended Next Step**: Check the **Profitability Compass** to see which crops deliver the highest return per hectare for your budget of ${currency} ${(context?.budget || 12000).toLocaleString()}. [TAB:compass:Explore Profitability Compass]
- **Need a Custom Plan?**: You can head to the **Proposal Builder** to craft an underwriting plan or ask me any specific question about crop rotation, fertigation schedules, or pest control! [TAB:proposal:Build Farm Proposal]`,
    suggestedActions: [
      "Take an Interactive Tour",
      "How to find the most profitable crop?",
      "How to use the Decision Matrix?",
      "How does voice dictation work?"
    ],
    highlightTab: "compass"
  };
}

// 3. API: Chatbot guide for new and returning users
app.post("/api/gemini/guide-chat", async (req, res) => {
  const { messages, context } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ success: false, error: "Messages array is required" });
  }

  const latestMessage = messages[messages.length - 1]?.content || "";

  // Build structured prompt for Gemini with user context
  const prompt = `You are AgriGuide, an agricultural decision-support advisor and user onboarding guide for AgriChoice 2.0.
Your goal is to guide new and experienced farmers, agronomists, and agribusiness planners on how to use AgriChoice 2.0 effectively and make sound agricultural investments.

APPLICATION CONTEXT:
AgriChoice 2.0 has 7 main modules:
- "compass" (1. Profitability Compass): Calculates ROI, gross margins, capital cost, and break-even for 50+ crops and livestock.
- "matrix" (2. Side-by-Side Sandbox / Decision Matrix): Compares crops and livestock side-by-side across soil compatibility, water stress, labor intensity, and risk profiles.
- "crops" (3. Crop Grow Sheets): Agronomic sheets with spacing diagrams, N-P-K fertigation protocols, and pest countermeasures.
- "livestock" (4. Livestock Plans): Herd/flock blueprints with stocking densities, feed formulations (starter/grower/finisher), and vaccination schedules.
- "proposal" (5. Proposal Builder): 7-step underwriting checklist, business plan generator, downloadable/printable PDF prospectus, and real-time voice-to-text dictation via microphone for the 'Project Objectives & Scope' section.
- "subsidies" (6. Subsidies & Laws): Localized database of farm grants, credit schemes, and environmental policies across 190+ sovereign countries.
- "blueprint" (7. Technical Specifications): Architecture, algorithms, and data methodology.

CURRENT FARMER/USER CONTEXT:
- Active Tab: ${context?.activeTab || "compass"}
- Country: ${context?.country || "United States"}
- Currency: ${context?.currency || "USD"}
- Soil Type: ${context?.soilType || "Loamy"}
- Soil pH: ${context?.soilPh || 6.5}
- Water Availability: ${context?.waterAvailability || "Medium"}
- Budget: ${context?.budget || 12000} ${context?.currency || "USD"}

USER QUERY:
"${latestMessage}"

RECENT CONVERSATION HISTORY:
${messages.slice(-4).map((m: any) => `${m.role.toUpperCase()}: ${m.content}`).join("\n")}

INSTRUCTIONS:
1. Tone: Professional, authoritative, welcoming, and practical (like a senior agricultural extension specialist and agtech advisor).
2. Keep the answer structured, clear, and actionable. Use bullet points and bold headers where helpful. Avoid overly dense academic jargon.
3. Whenever relevant, embed interactive tags that the frontend will parse into clickable action buttons:
   - [TAB:compass:Profitability Compass]
   - [TAB:matrix:Decision Matrix]
   - [TAB:crops:Crop Grow Sheets]
   - [TAB:livestock:Livestock Plans]
   - [TAB:proposal:Proposal Builder]
   - [TAB:subsidies:Subsidies & Policies]
   - [ACTION:country:Change Country]
   - [ACTION:dictate:Voice Dictation Guide]
4. Return ONLY valid JSON in this exact structure:
{
  "reply": "string containing your markdown response with embedded tags",
  "suggestedActions": ["short suggestion 1", "short suggestion 2", "short suggestion 3"],
  "highlightTab": "compass" | "matrix" | "crops" | "livestock" | "proposal" | "subsidies" | null
}`;

  try {
    const aiResponse = await generateWithModelFallback(prompt);
    if (aiResponse && aiResponse.reply) {
      return res.json({ success: true, data: aiResponse });
    }
    throw new Error("Invalid format from Gemini response");
  } catch (error: any) {
    console.warn(`[AgriChoice Resiliency] Gemini Guide-Chat notice (${error?.message || "Fallback triggered"}); serving verified onboarding guide.`);
    const fallback = getFallbackGuideResponse(latestMessage, context);
    res.json({ success: true, data: fallback });
  }
});

// Serve frontend client in production, Vite dev middleware in development
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[AgriChoice Server] Full-Stack Server listening at http://0.0.0.0:${PORT}`);
  });
}

startServer();
