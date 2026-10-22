"""Print the stored quotation leads (handy in the demo).  Run from the backend folder:  python show_leads.py"""
from app.leads import data_dir, list_leads


def main() -> None:
    rows = list_leads()
    print(f"Leads database: {data_dir() / 'leads.db'}")
    if not rows:
        print("No leads stored yet.")
        return
    for r in rows:
        print(
            f"\n#{r['id']}  {r['created_at']}\n"
            f"  {r['name']} ({r['company']})  |  {r['email']}  |  {r['phone']}\n"
            f"  application: {r['application']}  |  product: {r['recommended_product_id']}\n"
            f"  requirement: {r['requirement_notes']}\n"
            f"  follow-up call: {r['followup_at']}  |  email: {r['email_status']}"
        )


if __name__ == "__main__":
    main()