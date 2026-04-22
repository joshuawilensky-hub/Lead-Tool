"""
Enterprise SDR Prospecting Agent
AI-powered pipeline builder: Find → Research → Contacts → Pitch
Configurable for any SaaS product. Multi-provider LLM support with live web search.
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
import urllib.parse

st.set_page_config(page_title="Enterprise SDR Prospecting Agent", page_icon="🎯", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT CONFIGURATION — change this block to target any product
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_PRODUCT = {
    "name": "Notion",
    "tagline": "All-in-one connected workspace",
    "value_prop": "Replaces Confluence + Jira + Google Docs + Asana with one workspace. AI agents automate workflows, answer questions across the workspace, and connect to existing tools.",
    "pricing": "$18/user/month (Business), custom Enterprise pricing",
    "differentiator": "AI agents that handle entire workflows autonomously — the 2025-2026 platform differentiator",
    "competitors": "Confluence, Jira, Asana, Monday.com, ClickUp, Google Docs, SharePoint",
    "icp_employees_min": 500,
    "icp_employees_max": 3000,
    "icp_revenue_min": "$50M",
    "icp_revenue_max": "$500M",
    "segment": "Mid-Market",
}

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
LEAD_SOURCES = {
    "general": {
        "name": "General Search",
        "description": "Broad search by company size and vertical",
        "query": "companies {min} to {max} employees US headquarters 2024 2025"
    },
    "inc5000": {
        "name": "Inc 5000 / Deloitte Fast 500",
        "description": "Fastest-growing companies by revenue growth",
        "query": "Inc 5000 Deloitte Fast 500 fastest growing companies {min} to {max} employees 2024 2025"
    },
    "growth_stage": {
        "name": "Series C–E Startups",
        "description": "Growth-stage companies that recently raised and are scaling",
        "query": "Series C D E startups {min} to {max} employees 2024 2025 funding"
    },
    "hiring_signal": {
        "name": "Actively Hiring (Ops / IT)",
        "description": "Companies hiring for ops/productivity/IT roles — buying signal",
        "query": "companies hiring operations productivity IT {min} {max} employees 2024 2025"
    },
}

VERTICALS = {
    "saas_tech": "SaaS / Tech",
    "financial_services": "Financial Services / Fintech",
    "professional_services": "Professional Services / Consulting",
    "healthcare_biotech": "Healthcare / Biotech / Life Sciences",
    "media_creative": "Media / Creative / Agencies",
    "ecommerce_dtc": "E-commerce / DTC / Retail Tech",
    "education": "Education / EdTech",
}

PERSONAS = {
    "economic_buyer": {
        "label": "Economic Buyer",
        "description": "Signs the check. Cares about consolidation savings, security, reducing tool sprawl.",
        "titles": ["VP of IT", "Director of IT", "CTO", "VP Engineering", "CIO"],
    },
    "operational_champion": {
        "label": "Operational Champion",
        "description": "Feels the pain daily. Cares about team productivity, onboarding, reducing context-switching.",
        "titles": ["VP Operations", "Director of Operations", "Chief of Staff", "Head of Workplace", "COO"],
    },
    "end_user_champion": {
        "label": "End-User Champion",
        "description": "Already uses or would advocate for the product. Drives bottom-up adoption.",
        "titles": ["Head of Knowledge Management", "Director of Productivity", "Program Manager", "Head of PMO"],
    },
}


# ── System Prompts (templated for any product) ────────────────────────────────

def build_find_prompt(product):
    return f"""Find companies matching the given criteria. Return ONLY a JSON array:
[{{"company":"Name","website":"URL","employee_count":"number or range","estimated_revenue":"e.g. $80M or $100-200M or Unknown","vertical":"SaaS/Tech, Financial Services, Professional Services, Healthcare/Biotech, Media/Creative, E-commerce/DTC, Education, or Other"}}]
No markdown, no explanation, just the JSON array."""


def build_research_prompt(product):
    competitors = product["competitors"]
    return f"""Research a company's productivity/collaboration tool stack. Return ONLY this JSON:
{{"company":"Name","current_stack":{{"docs_wiki":"tool name or Unknown","project_mgmt":"tool name or Unknown","communication":"tool name or Unknown"}},"stack_fragmentation":"High (4+)/Medium (2-3)/Low/Unknown","growth_signal":"hiring/funding/expansion/reorg/None","product_adoption_signal":"Yes-describe/No/Unclear","pain_signals":"describe or None","competitor_tools":["list"],"tier":"Tier 1/Tier 2/Tier 3/Skip","tier_rationale":"one sentence","entry_point":"Bottom-up/Top-down/Expansion"}}

Context: We're selling {product['name']} ({product['tagline']}). Key competitors: {competitors}.
Tiers: 1=existing product usage OR competitor displacement opportunity. 2=high fragmentation+growth signals. 3=right size, no pain. Skip=wrong size or existing customer.
No markdown, just JSON."""


def build_contacts_prompt(product):
    return f"""Find contacts at a company matching requested personas. Return ONLY a JSON array:
[{{"company":"Name","name":"Full Name or Not Found","title":"Job title","persona":"Economic Buyer/Operational Champion/End-User Champion","seniority":"C-Suite/VP/Director/Head/Manager","why_relevant":"one sentence","linkedin_search_url":"LinkedIn search URL","confidence":"High/Medium/Low"}}]

Personas: Economic Buyer=VP IT/CTO/CIO. Operational Champion=VP Ops/Chief of Staff/COO. End-User Champion=Head of Knowledge Mgmt/Director of Productivity.
Search for real people. If name not found, set "Not Found" but still provide title and LinkedIn URL. No markdown, just JSON."""


def build_pitch_prompt(product):
    return f"""Generate a personalized sales pitch for {product['name']}. 
Product: {product['value_prop']}
Pricing: {product['pricing']}
Key differentiator: {product['differentiator']}

Return ONLY this JSON:
{{"company":"Name","contact_name":"Name","contact_title":"Title","persona":"Persona","email_subject":"max 8 words","opening_line":"personalized first sentence","pitch_angle":"one sentence value prop","talking_points":["1","2","3"],"consolidation_savings":"estimate or null","objection":"likely objection","objection_response":"how to handle","call_to_action":"next step"}}
No markdown, just JSON."""


# ══════════════════════════════════════════════════════════════════════════════
# API & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(provider, api_key, system, user, use_search=True):
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        kwargs = dict(model="claude-sonnet-4-5", max_tokens=4096, system=system,
                      messages=[{"role": "user", "content": user}])
        if use_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
        msg = client.messages.create(**kwargs)
        return "".join(b.text for b in msg.content if hasattr(b, "text"))
    elif provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        tools = []
        if use_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools if tools else None,
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=user, config=config,
        )
        return response.text or ""
    elif provider == "perplexity":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        r = client.chat.completions.create(model="sonar",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content
    return ""


def parse_json_array(text):
    s, e = text.find('['), text.rfind(']') + 1
    if s >= 0 and e > s:
        try: return json.loads(text[s:e])
        except: pass
    return []

def parse_json_object(text):
    s, e = text.find('{'), text.rfind('}') + 1
    if s >= 0 and e > s:
        try: return json.loads(text[s:e])
        except: pass
    return {}

def li_url(company, title):
    return f"https://www.linkedin.com/search/results/people/?keywords={urllib.parse.quote(f'{company} {title}')}"

def handle_error(e, context=""):
    msg = str(e)
    if "rate_limit" in msg or "429" in msg:
        st.caption(f"⏳ Rate limit hit{' — ' + context + ' skipped' if context else ''}. Wait ~60s.")
    else:
        st.error(f"Error: {msg[:120]}")

def highlight_tier(val):
    m = {'Tier 1': 'background-color:#DCFCE7;color:#166534', 'Tier 2': 'background-color:#FEF9C3;color:#854D0E',
         'Tier 3': 'background-color:#F3F4F6;color:#374151', 'Skip': 'background-color:#FEE2E2;color:#991B1B'}
    return m.get(val, '')

def highlight_confidence(val):
    m = {'High': 'background-color:#DCFCE7;color:#166534', 'Medium': 'background-color:#FEF9C3;color:#854D0E',
         'Low': 'background-color:#FEE2E2;color:#991B1B'}
    return m.get(val, '')


# ── Core Workflow Functions ───────────────────────────────────────────────────

def find_companies(provider, api_key, product, source_key, verticals=None, exclude=None, count=10):
    source = LEAD_SOURCES[source_key]
    query = source["query"].format(min=product["icp_employees_min"], max=product["icp_employees_max"])
    excl = f"\nDo NOT include these companies:\n{json.dumps(exclude[:200])}" if exclude else ""
    verts = ""
    if verticals:
        names = [VERTICALS[v] for v in verticals if v in VERTICALS]
        if names: verts = f"\nFocus ONLY on these verticals: {', '.join(names)}."

    prompt = f"""Search for: {query}

Find {count} US-based companies with {product['icp_employees_min']}–{product['icp_employees_max']} employees and ~{product['icp_revenue_min']}–{product['icp_revenue_max']} revenue.{verts}
Return a JSON array with company, website, employee_count, estimated_revenue, vertical.{excl}"""

    try:
        text = call_llm(provider, api_key, build_find_prompt(product), prompt, use_search=True)
        return parse_json_array(text)
    except Exception as e:
        handle_error(e, source["name"])
        return []


def research_company(provider, api_key, product, company_data):
    company = company_data.get("company", "Unknown")
    prompt = f"""Research {company}'s productivity tools. Website: {company_data.get('website','Unknown')}. Employees: {company_data.get('employee_count','Unknown')}. Vertical: {company_data.get('vertical','Unknown')}.
What do they use for docs/wiki, project management, communication? Any {product['name']} usage? Return JSON as specified."""

    try:
        text = call_llm(provider, api_key, build_research_prompt(product), prompt, use_search=True)
        result = parse_json_object(text)
        if result:
            result["company"] = company
        return result
    except Exception as e:
        handle_error(e, company)
        return {}


def find_contacts(provider, api_key, product, company_data, research_data, personas, quick_mode=True):
    company = company_data.get("company", "Unknown")
    stack = research_data.get("current_stack", {})
    count_per = 1 if quick_mode else 2
    persona_lines = []
    for p in personas:
        info = PERSONAS.get(p, {})
        persona_lines.append(f"- {info['label']}: titles like {', '.join(info['titles'][:3])}")

    prompt = f"""Company: {company}
Tech Stack: {json.dumps(stack)}
Tier: {research_data.get('tier', research_data.get('notion_tier', 'Unknown'))}
Vertical: {company_data.get('vertical', 'Unknown')}
Employees: {company_data.get('employee_count', 'Unknown')}

Find {len(personas) * count_per} contacts — {count_per} per persona:
{chr(10).join(persona_lines)}

Search for real people currently at this company."""

    try:
        text = call_llm(provider, api_key, build_contacts_prompt(product), prompt, use_search=True)
        contacts = parse_json_array(text)
        for c in contacts:
            c["company"] = company
            if not c.get("linkedin_search_url"):
                c["linkedin_search_url"] = li_url(company, c.get("title", ""))
        return contacts
    except Exception as e:
        handle_error(e, company)
        return []


def generate_pitch(provider, api_key, product, company_data, research_data, contact):
    prompt = f"""Company: {company_data.get('company', '')}
Stack: {json.dumps(research_data.get('current_stack', {}))}
Fragmentation: {research_data.get('stack_fragmentation', 'Unknown')}
Tier: {research_data.get('tier', research_data.get('notion_tier', 'Unknown'))}
Entry Point: {research_data.get('entry_point', 'Unknown')}

Contact: {contact.get('name', 'Unknown')} — {contact.get('title', '')} ({contact.get('persona', '')})
Why relevant: {contact.get('why_relevant', '')}

Generate a personalized pitch for this contact."""

    try:
        text = call_llm(provider, api_key, build_pitch_prompt(product), prompt, use_search=False)
        result = parse_json_object(text)
        if result:
            result["company"] = company_data.get("company", "")
            result["contact_name"] = contact.get("name", "")
            result["contact_title"] = contact.get("title", "")
            result["persona"] = contact.get("persona", "")
        return result
    except Exception as e:
        handle_error(e)
        return {}


# ── Pitch Card Renderer ──────────────────────────────────────────────────────

def render_pitch_card(p):
    if not p: return
    persona = p.get("persona", "")
    colors = {"Economic Buyer": "#2563EB", "Operational Champion": "#7C3AED", "End-User Champion": "#059669"}
    a = colors.get(persona, "#6B7280")

    st.markdown(f"""
    <div style="border:2px solid {a};border-radius:12px;padding:20px;margin-bottom:16px;
                background:linear-gradient(135deg,{a}08 0%,transparent 60%);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap;">
            <span style="background:{a};color:white;padding:4px 12px;border-radius:20px;font-size:12px;
                         font-weight:700;letter-spacing:.5px;">{persona.upper()}</span>
            <span style="font-size:15px;font-weight:700;color:#111827;">{p.get('contact_name','')}</span>
            <span style="font-size:13px;color:#6B7280;">{p.get('contact_title','')} at {p.get('company','')}</span>
        </div>
        <p style="font-size:14px;font-style:italic;color:#374151;margin:0;line-height:1.5;">"{p.get('pitch_angle','')}"</p>
    </div>""", unsafe_allow_html=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**✉️ Cold Email**")
        st.markdown(f"""<div style="background:#EFF6FF;border:1px solid #93C5FD;padding:12px;border-radius:8px;margin-bottom:12px;">
            <p style="font-size:12px;color:#1E40AF;margin:0 0 6px 0;"><strong>Subject:</strong> {p.get('email_subject','')}</p>
            <p style="font-size:13px;color:#1F2937;margin:0;line-height:1.5;">{p.get('opening_line','')}</p></div>""", unsafe_allow_html=True)
        st.markdown("**🎯 Talking Points**")
        for i, tp in enumerate(p.get("talking_points", []), 1):
            st.markdown(f"""<div style="background:#F8FAFC;border-left:3px solid {a};padding:8px 12px;margin-bottom:6px;
                        border-radius:0 8px 8px 0;font-size:13px;color:#374151;">
                <strong style="color:{a};">{i}.</strong> {tp}</div>""", unsafe_allow_html=True)
    with cr:
        if p.get("consolidation_savings"):
            st.markdown("**💰 Savings**")
            st.markdown(f"""<div style="background:#F0FDF4;border:1px solid #86EFAC;padding:10px;border-radius:8px;
                        font-size:13px;color:#166534;margin-bottom:12px;">{p['consolidation_savings']}</div>""", unsafe_allow_html=True)
        if p.get("objection"):
            st.markdown("**🛡️ Objection**")
            st.markdown(f"""<div style="background:#FEF2F2;border:1px solid #FCA5A5;padding:10px;border-radius:8px;margin-bottom:12px;">
                <p style="font-size:12px;color:#991B1B;margin:0 0 6px 0;"><strong>They'll say:</strong> "{p['objection']}"</p>
                <p style="font-size:12px;color:#1F2937;margin:0;"><strong>You say:</strong> {p.get('objection_response','')}</p></div>""", unsafe_allow_html=True)
        if p.get("call_to_action"):
            st.markdown("**🎬 CTA**")
            st.markdown(f"""<div style="background:#F5F3FF;border:1px solid #C4B5FD;padding:10px;border-radius:8px;
                        font-size:13px;color:#5B21B6;">{p['call_to_action']}</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Settings")

# Provider config
provider = st.sidebar.selectbox("LLM Provider", ["anthropic", "gemini", "perplexity", "openai"],
    help="Anthropic and Gemini include live web search. Gemini has generous free-tier rate limits.")
env_keys = {"perplexity": "PERPLEXITY_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
env_var = env_keys.get(provider, "")
default_key = ""
try:
    default_key = st.secrets.get(env_var, os.environ.get(env_var, ""))
except Exception:
    default_key = os.environ.get(env_var, "")
api_key = st.sidebar.text_input("API Key", value=default_key, type="password")

st.sidebar.markdown("---")

# Product configuration
st.sidebar.subheader("🏷️ Product Config")
with st.sidebar.expander("Customize target product", expanded=False):
    product_name = st.text_input("Product name", value=DEFAULT_PRODUCT["name"], key="pn")
    product_tagline = st.text_input("Tagline", value=DEFAULT_PRODUCT["tagline"], key="pt")
    product_value = st.text_area("Value prop", value=DEFAULT_PRODUCT["value_prop"], height=80, key="pv")
    product_pricing = st.text_input("Pricing", value=DEFAULT_PRODUCT["pricing"], key="pp")
    product_diff = st.text_input("Differentiator", value=DEFAULT_PRODUCT["differentiator"], key="pd")
    product_comp = st.text_input("Competitors", value=DEFAULT_PRODUCT["competitors"], key="pc")
    icp_min = st.number_input("ICP min employees", value=DEFAULT_PRODUCT["icp_employees_min"], key="imin")
    icp_max = st.number_input("ICP max employees", value=DEFAULT_PRODUCT["icp_employees_max"], key="imax")
    icp_rev_min = st.text_input("ICP min revenue", value=DEFAULT_PRODUCT["icp_revenue_min"], key="rmin")
    icp_rev_max = st.text_input("ICP max revenue", value=DEFAULT_PRODUCT["icp_revenue_max"], key="rmax")
    segment_label = st.text_input("Segment label", value=DEFAULT_PRODUCT["segment"], key="seg")

# Build active product config
product = {
    "name": product_name, "tagline": product_tagline, "value_prop": product_value,
    "pricing": product_pricing, "differentiator": product_diff, "competitors": product_comp,
    "icp_employees_min": icp_min, "icp_employees_max": icp_max,
    "icp_revenue_min": icp_rev_min, "icp_revenue_max": icp_rev_max, "segment": segment_label,
}

st.sidebar.markdown("---")
st.sidebar.caption(f"Targeting: **{product['name']}** · {product['segment']}")
st.sidebar.caption(f"ICP: {product['icp_employees_min']}–{product['icp_employees_max']} employees")

# ── Session State ─────────────────────────────────────────────────────────────
for k in ['found_companies', 'research_results', 'contacts', 'pitches', 'company_names_db']:
    if k not in st.session_state:
        st.session_state[k] = [] if k != 'research_results' else {}

def normalise(n): return n.strip().lower()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title(f"🎯 {product['segment']} SDR Prospecting Agent")
st.caption(f"AI-powered pipeline builder for **{product['name']}** · Find → Research → Contacts → Pitch")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Find", "🔬 Research", "👥 Contacts", "✉️ Pitch"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: FIND
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Find")
    st.markdown(f"Search for {product['segment'].lower()} companies ({product['icp_employees_min']}–{product['icp_employees_max']} employees, "
                f"~{product['icp_revenue_min']}–{product['icp_revenue_max']} revenue) matching the ICP.")

    with st.expander("📂 Upload exclusion list (optional)", expanded=False):
        st.caption("Upload a CSV of companies to exclude from results.")
        seed_file = st.file_uploader("Upload CSV", type=["csv"], key="seed")
        if seed_file:
            sdf = pd.read_csv(seed_file)
            col = next((c for c in sdf.columns if c.lower() in ['company','name','brand','organization']), sdf.columns[0])
            new = sdf[col].dropna().astype(str).tolist()
            exist = {normalise(n) for n in st.session_state['company_names_db']}
            added = [n for n in new if normalise(n) not in exist]
            st.session_state['company_names_db'].extend(added)
            st.success(f"✅ Added {len(added)} to exclusion list")

    st.markdown("---")

    st.subheader("Lead Sources")
    selected_sources = []
    src_cols = st.columns(4)
    for i, (k, s) in enumerate(LEAD_SOURCES.items()):
        with src_cols[i]:
            if st.checkbox(s["name"], help=s["description"]): selected_sources.append(k)

    st.subheader("Vertical Filters")
    st.caption("Narrow by industry — leave all unchecked for all verticals")
    selected_verticals = []
    vc = st.columns(4)
    for i, (k, label) in enumerate(VERTICALS.items()):
        with vc[i % 4]:
            if st.checkbox(label, key=f"v_{k}"): selected_verticals.append(k)

    db_ct = len(st.session_state['company_names_db'])
    if db_ct: st.info(f"🔒 Exclusion active — {db_ct} companies excluded.")

    st.markdown("---")
    speed1 = st.radio("Speed", ["⚡ Lightning (5)", "🚀 Quick (10)", "📊 Full (20)"],
                      horizontal=True, key="speed1")
    count_map = {"⚡ Lightning (5)": 5, "🚀 Quick (10)": 10, "📊 Full (20)": 20}
    find_count = count_map[speed1]

    if st.button("🔍 Find Companies", type="primary", disabled=not api_key or not selected_sources):
        all_r = []
        prog = st.progress(0)
        stat = st.empty()
        for i, sk in enumerate(selected_sources):
            stat.write(f"Searching {LEAD_SOURCES[sk]['name']}...")
            r = find_companies(provider, api_key, product, sk, selected_verticals,
                              st.session_state['company_names_db'], find_count)
            all_r.extend(r)
            prog.progress((i+1)/len(selected_sources))
            if i < len(selected_sources)-1: time.sleep(10)
        seen = set()
        deduped = []
        for r in all_r:
            n = normalise(r.get("company",""))
            if n and n not in seen:
                seen.add(n)
                deduped.append(r)
                st.session_state['company_names_db'].append(r.get("company",""))
        st.session_state['found_companies'] = deduped
        stat.write(f"✅ Found {len(deduped)} companies")

    if st.session_state['found_companies']:
        companies = st.session_state['found_companies']
        st.markdown("---")
        st.subheader(f"📋 Found Companies ({len(companies)})")
        rows = [{"Company": c.get("company",""), "Website": c.get("website",""),
                 "Employees": c.get("employee_count",""), "Est. Revenue": c.get("estimated_revenue",""),
                 "Vertical": c.get("vertical","")} for c in companies]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"Website": st.column_config.LinkColumn("Website")})
        st.download_button("📥 Export CSV", df.to_csv(index=False),
                          f"find_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", key="t1x")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Research")
    st.markdown("Deep-dive into each company's **tech stack, growth signals, product adoption, and tier ranking**.")

    companies = st.session_state.get('found_companies', [])

    st.subheader("Select Companies")
    input_method = st.radio("Input method", ["From Find tab", "Upload CSV", "Enter manually"],
                           horizontal=True, key="r_input")

    research_companies = []

    if input_method == "From Find tab":
        if not companies:
            st.info("No companies found yet. Run the **Find** tab first, or use CSV upload / manual entry.")
        else:
            opts = [c.get("company","") for c in companies]
            sel = st.multiselect("Choose companies to research", opts, default=opts[:5], key="r_sel")
            research_companies = [next((c for c in companies if c.get("company")==n), {"company":n}) for n in sel]

    elif input_method == "Upload CSV":
        csv_file = st.file_uploader("Upload CSV with company names", type=["csv"], key="r_csv")
        if csv_file:
            df_up = pd.read_csv(csv_file)
            col = next((c for c in df_up.columns if c.lower() in ['company','name','brand','organization']), df_up.columns[0])
            names = df_up[col].dropna().astype(str).tolist()
            research_companies = [{"company": n} for n in names]
            st.caption(f"{len(research_companies)} companies loaded from CSV")

    elif input_method == "Enter manually":
        manual = st.text_area("Company names (one per line)", placeholder="Lattice\nGong\nFigma", height=120, key="r_manual")
        if manual:
            names = [n.strip() for n in manual.split('\n') if n.strip()]
            research_companies = [{"company": n} for n in names]

    if research_companies:
        sel_names = [c.get("company","") for c in research_companies]
        st.markdown("---")
        st.caption(f"{len(sel_names)} companies selected — {len(sel_names)} API calls")

        if st.button("🔬 Research Companies", type="primary", disabled=not api_key or not sel_names):
            prog = st.progress(0)
            stat = st.empty()
            for i, name in enumerate(sel_names):
                stat.write(f"Researching {name} ({i+1}/{len(sel_names)})...")
                cdata = next((c for c in research_companies if c.get("company") == name), {"company": name})
                result = research_company(provider, api_key, product, cdata)
                if result:
                    st.session_state['research_results'][name] = result
                    if not any(c.get("company") == name for c in st.session_state['found_companies']):
                        st.session_state['found_companies'].append(cdata)
                prog.progress((i+1)/len(sel_names))
                if i < len(sel_names)-1: time.sleep(10)
            stat.write(f"✅ Researched {len(sel_names)} companies")

    research = st.session_state.get('research_results', {})
    if research:
        st.markdown("---")
        res_col, legend_col = st.columns([3, 1])

        with legend_col:
            st.markdown("#### Tier Scoring")
            st.markdown("""
            <div style="font-size:12px;line-height:1.8;">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    <span style="background:#DCFCE7;color:#166534;padding:2px 8px;border-radius:4px;font-weight:600;font-size:11px;">Tier 1</span>
                    <span>Existing adoption or competitor displacement</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    <span style="background:#FEF9C3;color:#854D0E;padding:2px 8px;border-radius:4px;font-weight:600;font-size:11px;">Tier 2</span>
                    <span>High fragmentation + growth signals</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    <span style="background:#F3F4F6;color:#374151;padding:2px 8px;border-radius:4px;font-weight:600;font-size:11px;">Tier 3</span>
                    <span>Right size, no clear pain signals</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="background:#FEE2E2;color:#991B1B;padding:2px 8px;border-radius:4px;font-weight:600;font-size:11px;">Skip</span>
                    <span>Wrong size or existing customer</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with res_col:
            st.subheader(f"📋 Research Results ({len(research)})")

            tier_filter = st.multiselect("Filter by Tier", ["Tier 1","Tier 2","Tier 3","Skip"],
                                         default=["Tier 1","Tier 2","Tier 3"], key="t2_tf")

            tiers = [v.get("tier", v.get("notion_tier","")) for v in research.values()]
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("🟢 Tier 1", tiers.count("Tier 1"))
            m2.metric("🟡 Tier 2", tiers.count("Tier 2"))
            m3.metric("🟠 Tier 3", tiers.count("Tier 3"))
            m4.metric("🔴 Skip", tiers.count("Skip"))

            r_rows = []
            for name, r in research.items():
                stack = r.get("current_stack", {})
                r_rows.append({
                    "Company": name,
                    "Docs/Wiki": stack.get("docs_wiki","") if isinstance(stack, dict) else "",
                    "Project Mgmt": stack.get("project_mgmt","") if isinstance(stack, dict) else "",
                    "Communication": stack.get("communication","") if isinstance(stack, dict) else "",
                    "Fragmentation": r.get("stack_fragmentation",""),
                    "Growth Signal": r.get("growth_signal",""),
                    "Adoption Signal": r.get("product_adoption_signal", r.get("notion_adoption_signal","")),
                    "Tier": r.get("tier", r.get("notion_tier","")),
                    "Entry Point": r.get("entry_point",""),
                    "Rationale": r.get("tier_rationale",""),
                })
            df_r = pd.DataFrame(r_rows)
            if tier_filter:
                df_r_display = df_r[df_r["Tier"].isin(tier_filter)]
            else:
                df_r_display = df_r

            tier_ord = {"Tier 1":0,"Tier 2":1,"Tier 3":2,"Skip":3}
            df_r_display = df_r_display.copy()
            df_r_display["_sort"] = df_r_display["Tier"].map(tier_ord)
            df_r_display = df_r_display.sort_values("_sort").drop(columns=["_sort"])

            styled = df_r_display.style.map(highlight_tier, subset=["Tier"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.download_button("📥 Export Research CSV", df_r.to_csv(index=False),
                              f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", key="t2x")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: CONTACTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Contacts")
    st.markdown("Find decision-makers and champions at researched companies.")

    research = st.session_state.get('research_results', {})
    companies = st.session_state.get('found_companies', [])

    if not research:
        st.info("No companies researched yet. Go to the **Research** tab first.")
    else:
        st.subheader("Persona Filters")
        sel_personas = []
        pc = st.columns(3)
        for i, (k, info) in enumerate(PERSONAS.items()):
            with pc[i]:
                if st.checkbox(info["label"], value=True, help=info["description"], key=f"p_{k}"):
                    sel_personas.append(k)
                st.caption(f"e.g. {', '.join(info['titles'][:3])}")

        if not sel_personas:
            st.warning("Select at least one persona.")

        st.subheader("Select Companies")
        researched_names = list(research.keys())
        tier_map = {n: research[n].get("tier", research[n].get("notion_tier","")) for n in researched_names}
        c_opts = [f"{n}  ({tier_map.get(n,'')})" for n in researched_names]
        sel_c = st.multiselect("Choose companies", c_opts, default=c_opts[:3], key="c_sel")
        sel_c_names = [o.rsplit("  (", 1)[0] for o in sel_c]

        st.markdown("---")
        qm3 = st.toggle("⚡ Quick Mode", value=True, key="qm3",
                         help="On: 1 contact/persona. Off: 2 contacts/persona.")
        cper = 1 if qm3 else 2
        st.caption(f"{len(sel_personas)} personas × {cper} contacts × {len(sel_c_names)} companies = "
                   f"~{len(sel_c_names)} API calls")

        if st.button("👥 Find Contacts", type="primary",
                     disabled=not api_key or not sel_c_names or not sel_personas):
            all_contacts = []
            prog = st.progress(0)
            stat = st.empty()
            for i, name in enumerate(sel_c_names):
                stat.write(f"Finding contacts at {name} ({i+1}/{len(sel_c_names)})...")
                cdata = next((c for c in companies if c.get("company") == name), {"company": name})
                rdata = research.get(name, {})
                cts = find_contacts(provider, api_key, product, cdata, rdata, sel_personas, qm3)
                all_contacts.extend(cts)
                prog.progress((i+1)/len(sel_c_names))
                if i < len(sel_c_names)-1: time.sleep(10)
            st.session_state['contacts'] = all_contacts
            stat.write(f"✅ Found {len(all_contacts)} contacts")

    if st.session_state.get('contacts'):
        contacts = st.session_state['contacts']
        st.markdown("---")
        st.subheader(f"📋 Contacts ({len(contacts)})")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            pf = st.multiselect("Filter by Persona",
                list({c.get("persona","") for c in contacts}),
                default=list({c.get("persona","") for c in contacts}), key="t3_pf")
        with fc2:
            cf = st.multiselect("Filter by Company",
                sorted({c.get("company","") for c in contacts}), key="t3_cf")
        with fc3:
            conf_f = st.multiselect("Filter by Confidence", ["High","Medium","Low"],
                default=["High","Medium","Low"], key="t3_conf")

        c_rows = [{"Company": c.get("company",""), "Name": c.get("name",""), "Title": c.get("title",""),
                   "Persona": c.get("persona",""), "Seniority": c.get("seniority",""),
                   "Why Relevant": c.get("why_relevant",""),
                   "Confidence": c.get("confidence","").split(" ")[0] if c.get("confidence") else "",
                   "LinkedIn": c.get("linkedin_search_url","")} for c in contacts]
        df_c = pd.DataFrame(c_rows)

        filtered = df_c.copy()
        if pf: filtered = filtered[filtered["Persona"].isin(pf)]
        if cf: filtered = filtered[filtered["Company"].isin(cf)]
        if conf_f: filtered = filtered[filtered["Confidence"].isin(conf_f)]

        pm1,pm2,pm3 = st.columns(3)
        pm1.metric("💰 Economic Buyers", len(df_c[df_c["Persona"]=="Economic Buyer"]))
        pm2.metric("⚙️ Op Champions", len(df_c[df_c["Persona"]=="Operational Champion"]))
        pm3.metric("🚀 End-User Champions", len(df_c[df_c["Persona"]=="End-User Champion"]))

        styled = filtered.style.map(highlight_confidence, subset=["Confidence"])
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     column_config={"LinkedIn": st.column_config.LinkColumn("LinkedIn", display_text="🔗 Search")})

        st.download_button("📥 Export Contacts CSV", df_c.to_csv(index=False),
                          f"contacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", key="t3x")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PITCH
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Pitch")
    st.markdown("Generate personalized outreach using data from all prior tabs.")

    contacts = st.session_state.get('contacts', [])
    research = st.session_state.get('research_results', {})
    companies = st.session_state.get('found_companies', [])

    if not contacts:
        st.info("No contacts found yet. Go to the **Contacts** tab first.")
    else:
        st.subheader("Select Contacts to Pitch")

        sel_mode = st.radio("Selection mode", ["By Company (all contacts)", "Individual Contacts"],
                           horizontal=True)

        sel_contacts = []
        if sel_mode == "By Company (all contacts)":
            cos = sorted({c.get("company","") for c in contacts})
            sel_cos = st.multiselect("Select companies", cos, key="p_cos")
            sel_contacts = [c for c in contacts if c.get("company") in sel_cos]
            if sel_contacts:
                st.caption(f"{len(sel_contacts)} contacts across {len(sel_cos)} companies")
        else:
            p_pf = st.multiselect("Filter by Persona",
                ["Economic Buyer","Operational Champion","End-User Champion"],
                default=["Economic Buyer","Operational Champion","End-User Champion"], key="p_pf")
            labels = {}
            for c in contacts:
                if c.get("persona") in p_pf:
                    l = f"{c.get('name','?')} — {c.get('title','')} at {c.get('company','')} ({c.get('persona','')})"
                    labels[l] = c
            sel_labels = st.multiselect("Select contacts", list(labels.keys()), key="p_sel")
            sel_contacts = [labels[l] for l in sel_labels]

        if sel_contacts:
            st.caption(f"{len(sel_contacts)} pitch(es) — no web search, faster")
            if st.button("✉️ Generate Pitches", type="primary", disabled=not api_key):
                all_p = []
                prog = st.progress(0)
                stat = st.empty()
                for i, ct in enumerate(sel_contacts):
                    cn = ct.get("company","")
                    stat.write(f"Pitching {ct.get('name','?')} at {cn} ({i+1}/{len(sel_contacts)})...")
                    cdata = next((c for c in companies if c.get("company")==cn), {"company":cn})
                    rdata = research.get(cn, {})
                    pitch = generate_pitch(provider, api_key, product, cdata, rdata, ct)
                    if pitch: all_p.append(pitch)
                    prog.progress((i+1)/len(sel_contacts))
                    if i < len(sel_contacts)-1: time.sleep(5)
                st.session_state['pitches'].extend(all_p)
                stat.write(f"✅ Generated {len(all_p)} pitches")

    if st.session_state.get('pitches'):
        pitches = st.session_state['pitches']
        st.markdown("---")
        st.subheader(f"📋 Generated Pitches ({len(pitches)})")

        pf1, pf2 = st.columns(2)
        with pf1:
            p_cf = st.multiselect("Filter by Company",
                sorted({p.get("company","") for p in pitches}), key="t4_cf")
        with pf2:
            p_pf2 = st.multiselect("Filter by Persona",
                sorted({p.get("persona","") for p in pitches}),
                default=sorted({p.get("persona","") for p in pitches}), key="t4_pf")

        fp = pitches
        if p_cf: fp = [p for p in fp if p.get("company") in p_cf]
        if p_pf2: fp = [p for p in fp if p.get("persona") in p_pf2]

        for pitch in fp:
            render_pitch_card(pitch)
            st.markdown("")

        exp = [{"Company":p.get("company",""),"Contact":p.get("contact_name",""),
                "Title":p.get("contact_title",""),"Persona":p.get("persona",""),
                "Subject":p.get("email_subject",""),"Opening":p.get("opening_line",""),
                "Angle":p.get("pitch_angle",""),
                "TP1":p.get("talking_points",[""])[0] if p.get("talking_points") else "",
                "TP2":p.get("talking_points",["",""])[1] if len(p.get("talking_points",[]))>1 else "",
                "TP3":p.get("talking_points",["","",""])[2] if len(p.get("talking_points",[]))>2 else "",
                "Savings":p.get("consolidation_savings",""),
                "Objection":p.get("objection",""),"Response":p.get("objection_response",""),
                "CTA":p.get("call_to_action","")} for p in pitches]
        st.download_button("📥 Export Pitches CSV", pd.DataFrame(exp).to_csv(index=False),
                          f"pitches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", key="t4x")
