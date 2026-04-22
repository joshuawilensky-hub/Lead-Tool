# Enterprise SDR Prospecting Agent

AI-powered pipeline builder with a four-step workflow: **Find → Research → Contacts → Pitch**

Configurable for any SaaS product. Set your product name, value prop, competitors, and ICP via the sidebar config panel.

## What It Does

Each step uses LLM + live web search to build qualified pipeline from scratch:

1. **Find** — Searches for companies matching your ICP by size, revenue, vertical, and growth signals
2. **Research** — Deep-dives each company's tech stack, growth signals, existing adoption, and assigns tier priority
3. **Contacts** — Finds real decision-makers mapped to buying personas (Economic Buyer, Operational Champion, End-User Champion)
4. **Pitch** — Generates personalized outreach with email copy, talking points, objection handling, and CTAs

## Providers

| Provider | Web Search | Cost/Lead | Notes |
|---|---|---|---|
| Anthropic (Claude) | ✅ Native | ~$0.015 | Best structured output |
| Gemini | ✅ Native | ~$0.010 | Generous free tier |
| Perplexity (Sonar) | ✅ Native | ~$0.005 | Cheapest |
| OpenAI (GPT-4o) | ❌ | ~$0.025 | No live search |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Product Config

The sidebar expander lets you reconfigure the target product on the fly — product name, value prop, pricing, competitors, ICP size range. All prompts dynamically rebuild from the config.

## Built By

Josh Wilensky — [LinkedIn](https://www.linkedin.com/in/joshwilensky/)
