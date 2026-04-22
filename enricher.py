#!/usr/bin/env python3
"""
Notion Mid-Market SDR Prospecting Agent - CLI Enricher
Multi-Provider Edition with configurable waterfall fallback.

Targets organizations with 500–3,000 employees for net-new Notion business acquisition.

Providers supported:
- perplexity: Sonar with native search ($0.005-0.01/lead) - CHEAPEST
- anthropic: Claude with web search ($0.01-0.02/lead)
- openai: GPT-4o with web search ($0.02-0.03/lead)
- gemini: Gemini 1.5 Pro ($0.010/lead)

Usage:
    python enricher.py                           # Interactive mode, uses config
    python enricher.py companies.csv             # Process CSV
    python enricher.py --provider perplexity     # Use specific provider
    python enricher.py --config                  # Edit API keys and settings

Environment variables (or use --config to set):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    PERPLEXITY_API_KEY
    GEMINI_API_KEY
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm

console = Console()

CONFIG_FILE = Path.home() / ".notion_prospector_config.json"

DEFAULT_CONFIG = {
    "providers": {
        "anthropic": {
            "api_key": "",
            "model": "claude-sonnet-4-5",
            "cost_per_lead": 0.015,
            "enabled": True
        },
        "perplexity": {
            "api_key": "",
            "model": "sonar",
            "cost_per_lead": 0.005,
            "enabled": True
        },
        "openai": {
            "api_key": "",
            "model": "gpt-4o",
            "cost_per_lead": 0.025,
            "enabled": True
        },
        "gemini": {
            "api_key": "",
            "model": "gemini-1.5-pro",
            "cost_per_lead": 0.010,
            "enabled": True
        }
    },
    "waterfall_order": ["perplexity", "anthropic", "openai", "gemini"],
    "default_provider": "perplexity"
}

SYSTEM_PROMPT = """You are a deep-research assistant for a Notion Mid-Market SDR (Sales Development Representative) focused on net-new business acquisition targeting organizations with 500–3,000 employees.

Notion is an all-in-one workspace that replaces multiple tools (Confluence, Jira, Google Docs, Asana, Monday.com, Slack channels used for docs) with a single connected platform. Notion AI agents can automate workflows, answer questions across the workspace, and connect to existing tools — this is Notion's major 2025-2026 differentiator.

Your job is to research each company thoroughly and return a structured JSON object to help the SDR prioritize and prepare outreach.

For each company, search the web and return ONLY this JSON object with no other text:
{
    "company": "Company Name",
    "website": "official website URL",
    "employee_count": "estimated headcount (number or range)",
    "segment": "SaaS/Tech or Professional Services or Financial Services or Healthcare/Biotech or Media/Creative or Education or E-commerce/Retail or Other",
    "growth_signal": "hiring or funding or expansion or reorg or None",
    "current_stack": {
        "docs_wiki": "Confluence / Google Docs / SharePoint / Notion / Other / Unknown",
        "project_mgmt": "Asana / Monday / Jira / ClickUp / Notion / Other / Unknown",
        "communication": "Slack / Teams / Other / Unknown",
        "notes_knowledge": "Confluence / Notion / Coda / Slite / Other / Unknown"
    },
    "notion_adoption_signal": "Yes - describe (e.g. job postings mention Notion, engineering blog references it) / No / Unclear",
    "stack_fragmentation_score": "High (4+ tools) / Medium (2-3 tools) / Low (consolidated)",
    "pain_signals": "describe any signals of productivity pain - rapid hiring, remote/hybrid, recent M&A, tool consolidation initiatives, or None",
    "competitor_tools": ["list of direct Notion competitors in use"],
    "notion_tier": "Tier 1 / Tier 2 / Tier 3 / Skip",
    "tier_rationale": "one sentence explanation",
    "entry_point": "Bottom-up (already some Notion users) / Top-down (exec pitch to consolidate) / Expansion (using Notion free/Plus, upgrade to Business/Enterprise)",
    "recommended_contacts": "titles to target e.g. VP Operations, Head of IT, CTO, Chief of Staff",
    "linkedin_search_url": "https://www.linkedin.com/search/results/people/?keywords=COMPANY%20VP%20Operations%20Head%20of%20IT",
    "notes": "anything else useful for the SDR"
}

Tier logic - assign tiers as follows:
- Tier 1: Already has some Notion adoption (job posts, blog mentions, Glassdoor reviews reference Notion) BUT no enterprise contract — land-and-expand opportunity. OR: Uses Confluence + Jira and company is 500-3000 employees with growth/hiring signals — competitive displacement.
- Tier 2: 500-3000 employees, high stack fragmentation (4+ productivity tools), growth signals present. Greenfield opportunity.
- Tier 3: Right size but no clear tool pain or buying signals. Needs nurture.
- Skip: Under 500 or over 3000 employees, or already a known Notion Enterprise customer.

Return ONLY the JSON object. No markdown, no explanation, no code fences."""


def load_config() -> dict:
    """Load config from file or create default."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                for provider, settings in DEFAULT_CONFIG["providers"].items():
                    if provider not in config.get("providers", {}):
                        config.setdefault("providers", {})[provider] = settings
                return config
        except:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """Save config to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    console.print(f"[green]Config saved to {CONFIG_FILE}[/green]")


def configure_interactive():
    """Interactive configuration of API keys and settings."""
    config = load_config()

    console.print(Panel.fit(
        "[bold cyan]Notion Mid-Market Prospector Configuration[/bold cyan]\n\n"
        "Configure your API keys and provider preferences.",
        title="Settings"
    ))

    # Show current status
    console.print("\n[bold]Current Provider Status:[/bold]")
    table = Table(show_header=True)
    table.add_column("Provider")
    table.add_column("API Key")
    table.add_column("Cost/Lead")
    table.add_column("Enabled")

    for name, settings in config["providers"].items():
        key_status = "✓ Set" if settings.get("api_key") else "✗ Not set"
        key_style = "green" if settings.get("api_key") else "red"
        table.add_row(
            name,
            f"[{key_style}]{key_status}[/{key_style}]",
            f"${settings.get('cost_per_lead', 0):.3f}",
            "Yes" if settings.get("enabled") else "No"
        )
    console.print(table)

    # Configure each provider
    console.print("\n[bold]Configure Providers:[/bold]")
    for name in ["perplexity", "anthropic", "openai", "gemini"]:
        settings = config["providers"].get(name, {})

        if Confirm.ask(f"\nConfigure {name}?", default=False):
            current_key = settings.get("api_key", "")
            masked = f"...{current_key[-8:]}" if current_key else "not set"
            new_key = Prompt.ask(
                f"  API Key (current: {masked})",
                default="",
                password=True
            )
            if new_key:
                config["providers"][name]["api_key"] = new_key

            config["providers"][name]["enabled"] = Confirm.ask(
                f"  Enable {name}?",
                default=settings.get("enabled", True)
            )

    # Waterfall order
    console.print("\n[bold]Waterfall Order:[/bold]")
    console.print("  Providers are tried in this order if one fails.")
    console.print(f"  Current: {' → '.join(config.get('waterfall_order', []))}")

    if Confirm.ask("Change waterfall order?", default=False):
        console.print("  Enter provider names in order (comma-separated):")
        console.print("  Available: perplexity, anthropic, openai, gemini")
        order_input = Prompt.ask("  Order", default="perplexity,anthropic,openai,gemini")
        config["waterfall_order"] = [p.strip() for p in order_input.split(",")]

    config["default_provider"] = Prompt.ask(
        "\nDefault provider",
        choices=["perplexity", "anthropic", "openai", "gemini"],
        default=config.get("default_provider", "perplexity")
    )

    save_config(config)

    console.print("\n[bold green]Configuration complete![/bold green]")
    console.print(f"Default provider: {config['default_provider']}")
    console.print(f"Waterfall order: {' → '.join(config['waterfall_order'])}")


def get_api_key(provider: str, config: dict) -> Optional[str]:
    """Get API key from config or environment."""
    key = config.get("providers", {}).get(provider, {}).get("api_key")
    if key:
        return key

    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "gemini": "GEMINI_API_KEY"
    }
    return os.environ.get(env_map.get(provider, ""))


def enrich_with_anthropic(company_name: str, api_key: str, model: str) -> dict:
    """Research company using Claude with web search."""
    try:
        from anthropic import Anthropic
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
        from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f'Research "{company_name}" for Notion mid-market SDR prospecting. Find their tech stack, employee count, growth signals, and any existing Notion usage. Return only JSON.'
        }]
    )

    text_content = ""
    for block in message.content:
        if hasattr(block, 'text'):
            text_content += block.text

    return parse_json_response(text_content, company_name)


def enrich_with_openai(company_name: str, api_key: str, model: str) -> dict:
    """Research company using OpenAI with web search."""
    try:
        from openai import OpenAI
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
        from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Research "{company_name}" for Notion mid-market SDR prospecting. Find their tech stack, employee count, growth signals, and any existing Notion usage. Return only JSON.'}
        ]
    )

    text_content = response.choices[0].message.content
    return parse_json_response(text_content, company_name)


def enrich_with_perplexity(company_name: str, api_key: str, model: str) -> dict:
    """Research company using Perplexity Sonar (search-native, cheapest)."""
    try:
        from openai import OpenAI
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
        from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Research "{company_name}" for Notion mid-market SDR prospecting. Search for their employee count, tech stack (Confluence, Jira, Asana, Monday.com, Google Workspace, Slack, Notion), growth signals, and any existing Notion adoption. Return only JSON.'}
        ]
    )

    text_content = response.choices[0].message.content
    return parse_json_response(text_content, company_name)


def enrich_with_gemini(company_name: str, api_key: str, model: str) -> dict:
    """Research company using Google Gemini."""
    try:
        import google.generativeai as genai
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai", "-q"])
        import google.generativeai as genai

    genai.configure(api_key=api_key)

    llm = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT
    )

    response = llm.generate_content(f'Research "{company_name}" for Notion mid-market SDR prospecting. Find their tech stack, employee count, growth signals, and any existing Notion usage. Return only JSON.')

    return parse_json_response(response.text, company_name)


def parse_json_response(text: str, company_name: str) -> dict:
    """Parse JSON from LLM response."""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass

    return {
        "company": company_name,
        "website": "",
        "employee_count": "Error",
        "segment": "Error",
        "notion_tier": "Error",
        "stack_fragmentation_score": "Error",
        "notion_adoption_signal": "Error",
        "entry_point": "Error",
        "notes": f"Parse error: {text[:100]}"
    }


def enrich_company(company_name: str, config: dict, provider: Optional[str] = None) -> tuple:
    """
    Research a company using configured provider(s).
    Returns (result_dict, provider_used)
    """
    if provider:
        providers_to_try = [provider]
    else:
        providers_to_try = config.get("waterfall_order", ["perplexity", "anthropic", "openai", "gemini"])

    for prov in providers_to_try:
        settings = config.get("providers", {}).get(prov, {})

        if not settings.get("enabled", True):
            continue

        api_key = get_api_key(prov, config)
        if not api_key:
            continue

        model = settings.get("model", "")

        try:
            if prov == "anthropic":
                result = enrich_with_anthropic(company_name, api_key, model)
            elif prov == "openai":
                result = enrich_with_openai(company_name, api_key, model)
            elif prov == "perplexity":
                result = enrich_with_perplexity(company_name, api_key, model)
            elif prov == "gemini":
                result = enrich_with_gemini(company_name, api_key, model)
            else:
                continue

            if result.get("employee_count") != "Error":
                return result, prov

        except Exception as e:
            console.print(f"[yellow]  {prov} failed: {str(e)[:50]}[/yellow]")
            continue

    return {
        "company": company_name,
        "website": "",
        "employee_count": "Error",
        "segment": "Error",
        "notion_tier": "Error",
        "stack_fragmentation_score": "Error",
        "notion_adoption_signal": "Error",
        "entry_point": "Error",
        "notes": "All providers failed"
    }, "none"


def generate_linkedin_url(company_name: str) -> str:
    """Generate LinkedIn search URL for target contacts."""
    import urllib.parse
    query = f"{company_name} VP Operations Head of IT CTO Chief of Staff Director Productivity"
    return f"https://www.linkedin.com/search/results/people/?keywords={urllib.parse.quote(query)}"


def display_results(results: list, providers_used: dict):
    """Display results in a formatted table."""
    table = Table(title="Notion Mid-Market Prospecting Results", show_lines=True)

    table.add_column("Company", style="cyan", no_wrap=True)
    table.add_column("Employees", width=10)
    table.add_column("Segment", width=18)
    table.add_column("Fragmentation", width=14)
    table.add_column("Notion Signal", width=14)
    table.add_column("Tier", width=8)
    table.add_column("Entry Point", width=16)
    table.add_column("Source", width=10, style="dim")

    tier_colors = {
        "Tier 1": "green",
        "Tier 2": "yellow",
        "Tier 3": "dim",
        "Skip": "red",
        "Error": "red"
    }

    for r in results:
        tier = r.get("notion_tier", "Unknown")
        tier_style = tier_colors.get(tier, "white")
        company = r.get("company", "")

        # Truncate notion adoption signal for display
        notion_signal = r.get("notion_adoption_signal", "")
        if len(str(notion_signal)) > 14:
            notion_signal = str(notion_signal)[:12] + "…"

        table.add_row(
            company,
            str(r.get("employee_count", ""))[:10],
            r.get("segment", "")[:18],
            r.get("stack_fragmentation_score", "")[:14],
            str(notion_signal)[:14],
            f"[{tier_style}]{tier}[/{tier_style}]",
            r.get("entry_point", "")[:16],
            providers_used.get(company, "")[:10]
        )

    console.print(table)

    # Summary
    t1 = sum(1 for r in results if r.get("notion_tier") == "Tier 1")
    t2 = sum(1 for r in results if r.get("notion_tier") == "Tier 2")
    t3 = sum(1 for r in results if r.get("notion_tier") == "Tier 3")
    skip = sum(1 for r in results if r.get("notion_tier") == "Skip")

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  [green]Tier 1 (land-expand / displacement):[/green] {t1}")
    console.print(f"  [yellow]Tier 2 (greenfield):[/yellow] {t2}")
    console.print(f"  [dim]Tier 3 (nurture):[/dim] {t3}")
    console.print(f"  [red]Skip:[/red] {skip}")

    # Cost estimate
    provider_counts = {}
    for p in providers_used.values():
        provider_counts[p] = provider_counts.get(p, 0) + 1

    config = load_config()
    total_cost = 0
    for prov, count in provider_counts.items():
        cost = config.get("providers", {}).get(prov, {}).get("cost_per_lead", 0.01)
        total_cost += cost * count

    console.print(f"\n[dim]Estimated API cost: ${total_cost:.2f}[/dim]")


def save_results(results: list, output_path: str):
    """Save results to CSV file."""
    fieldnames = [
        "Company", "Website", "Employees", "Segment", "Tier",
        "Entry Point", "Fragmentation", "Notion Signal",
        "Growth Signal", "Competitor Tools", "Recommended Contacts",
        "LinkedIn Search", "Tier Rationale", "Notes"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            competitor_tools = r.get("competitor_tools", [])
            if isinstance(competitor_tools, list):
                competitor_tools = ", ".join(competitor_tools)

            writer.writerow({
                "Company": r.get("company", ""),
                "Website": r.get("website", ""),
                "Employees": r.get("employee_count", ""),
                "Segment": r.get("segment", ""),
                "Tier": r.get("notion_tier", ""),
                "Entry Point": r.get("entry_point", ""),
                "Fragmentation": r.get("stack_fragmentation_score", ""),
                "Notion Signal": r.get("notion_adoption_signal", ""),
                "Growth Signal": r.get("growth_signal", ""),
                "Competitor Tools": competitor_tools,
                "Recommended Contacts": r.get("recommended_contacts", ""),
                "LinkedIn Search": generate_linkedin_url(r.get("company", "")),
                "Tier Rationale": r.get("tier_rationale", ""),
                "Notes": r.get("notes", "")
            })

    console.print(f"\n[green]✓ Results saved to:[/green] {output_path}")


def load_companies_from_csv(filepath: str) -> list:
    """Load company names from a CSV file."""
    companies = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                if row[0].lower() in ['company', 'name', 'brand', 'organization', 'org']:
                    continue
                companies.append(row[0].strip())
    return companies


def process_companies(companies: list, config: dict, provider: Optional[str] = None):
    """Process a list of companies and display/save results."""
    results = []
    providers_used = {}

    if provider:
        console.print(f"[cyan]Using provider: {provider}[/cyan]")
    else:
        console.print(f"[cyan]Waterfall: {' → '.join(config.get('waterfall_order', []))}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Researching companies...", total=len(companies))

        for company in companies:
            progress.update(task, description=f"Researching: {company}")
            result, prov_used = enrich_company(company, config, provider)
            results.append(result)
            providers_used[company] = prov_used
            progress.advance(task)

    console.print()
    display_results(results, providers_used)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"notion_prospects_{timestamp}.csv"
    save_results(results, output_path)

    # Show top priorities
    tier1 = [r for r in results if r.get("notion_tier") == "Tier 1"]
    if tier1:
        console.print(f"\n[bold green]🎯 Top targets (Tier 1):[/bold green]")
        for r in tier1:
            console.print(f"  • {r['company']}: {r.get('entry_point', '')} — {r.get('tier_rationale', '')[:80]}")


def interactive_mode(config: dict, provider: Optional[str] = None):
    """Run in interactive mode."""
    console.print(Panel.fit(
        "[bold cyan]Notion Mid-Market Prospector[/bold cyan]\n\n"
        "Enter company names to research for Notion sales prospecting.\n"
        "Type 'done' when finished, 'quit' to exit.",
        title="Welcome"
    ))

    # Show provider status
    available = []
    for name, settings in config.get("providers", {}).items():
        if settings.get("enabled") and get_api_key(name, config):
            available.append(f"{name} (${settings.get('cost_per_lead', 0):.3f}/lead)")

    if not available:
        console.print("[red]No providers configured! Run with --config to set up.[/red]")
        return

    console.print(f"[dim]Available providers: {', '.join(available)}[/dim]")

    companies = []
    console.print("\n[bold]Enter company names (one per line, 'done' to process):[/bold]")

    while True:
        try:
            company = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if company.lower() == 'quit':
            return
        if company.lower() == 'done':
            break
        if company:
            companies.append(company)

    if companies:
        process_companies(companies, config, provider)


def main():
    parser = argparse.ArgumentParser(
        description="Notion Mid-Market SDR Prospecting Agent - Multi-Provider Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python enricher.py                              # Interactive mode
    python enricher.py companies.csv                # Process CSV
    python enricher.py --provider perplexity        # Use specific provider
    python enricher.py --config                     # Configure API keys

Providers (by cost):
    perplexity   $0.005/lead   Cheapest, search-native
    anthropic    $0.015/lead   Claude with web search
    openai       $0.025/lead   GPT-4o with web search
    gemini       $0.010/lead   Gemini Pro
        """
    )
    parser.add_argument('input', nargs='*', help='CSV file or company names')
    parser.add_argument('--provider', '-p', choices=['anthropic', 'openai', 'perplexity', 'gemini'],
                        help='Use specific provider instead of waterfall')
    parser.add_argument('--config', '-c', action='store_true',
                        help='Configure API keys and settings')
    parser.add_argument('--show-config', action='store_true',
                        help='Show current configuration')

    args = parser.parse_args()

    if args.config:
        configure_interactive()
        return

    config = load_config()

    if args.show_config:
        console.print(json.dumps(config, indent=2, default=str))
        return

    # Check if any provider is available
    has_provider = False
    for name in config.get("waterfall_order", []):
        if get_api_key(name, config) and config.get("providers", {}).get(name, {}).get("enabled"):
            has_provider = True
            break

    if not has_provider:
        console.print("[yellow]No API keys configured.[/yellow]")
        console.print("Run [bold]python enricher.py --config[/bold] to set up providers.")
        console.print("\nOr set environment variables:")
        console.print("  export PERPLEXITY_API_KEY=your_key  # Cheapest")
        console.print("  export ANTHROPIC_API_KEY=your_key")
        console.print("  export OPENAI_API_KEY=your_key")
        console.print("  export GEMINI_API_KEY=your_key")
        return

    # Process input
    if not args.input:
        interactive_mode(config, args.provider)
    elif len(args.input) == 1 and args.input[0].endswith('.csv'):
        filepath = args.input[0]
        if not Path(filepath).exists():
            console.print(f"[red]File not found: {filepath}[/red]")
            sys.exit(1)
        companies = load_companies_from_csv(filepath)
        console.print(f"[cyan]Loaded {len(companies)} companies from {filepath}[/cyan]")
        process_companies(companies, config, args.provider)
    elif len(args.input) == 1 and ',' in args.input[0]:
        companies = [c.strip() for c in args.input[0].split(',') if c.strip()]
        process_companies(companies, config, args.provider)
    else:
        companies = [c.strip() for c in args.input if c.strip()]
        process_companies(companies, config, args.provider)


if __name__ == "__main__":
    main()
