"""
Mock Core Banking Back-Office Application
Deliberately "legacy" UI: server-rendered HTML, table-based layout,
no id/data-testid/semantic attributes on interactive elements.
Used as the automation target in the take-home assignment.
"""
import time
from datetime import date
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "mock-secret-not-for-prod"

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

MEMBERS = {
    "12345": {
        "name": "Alice Nguyen",
        "member_id": "12345",
        "status": "Active",
        "accounts": [
            {"type": "Checking", "number": "****4521", "balance": 2_318.47},
            {"type": "Savings",  "number": "****7890", "balance": 5_432.10},
        ],
    },
    "67890": {
        "name": "Robert Kim",
        "member_id": "67890",
        "status": "Active",
        "accounts": [
            {"type": "Checking", "number": "****1100", "balance": 812.00},
        ],
    },
}

# Member IDs whose sub-account creation is locked to manager role only
PERMISSION_DENIED_MEMBERS = {"67890"}

SUB_ACCOUNT_TYPES = ["Money Market", "Holiday Club", "IRA", "Certificate of Deposit"]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("search"))


@app.route("/search", methods=["GET", "POST"])
def search():
    error = None
    if request.method == "POST":
        member_id = request.form.get("member_id", "").strip()
        if not member_id:
            error = "Please enter a Member ID."
        elif member_id not in MEMBERS:
            return redirect(url_for("not_found", member_id=member_id))
        else:
            return redirect(url_for("member_detail", member_id=member_id))
    return render_template("search.html", error=error)


@app.route("/member/<member_id>")
def member_detail(member_id):
    member = MEMBERS.get(member_id)
    if not member:
        return redirect(url_for("not_found", member_id=member_id))
    return render_template("member_detail.html", member=member)


@app.route("/member/<member_id>/balance")
def view_balance(member_id):
    member = MEMBERS.get(member_id)
    if not member:
        return redirect(url_for("not_found", member_id=member_id))
    account_type = request.args.get("type", "Savings")
    savings = next(
        (a for a in member["accounts"] if a["type"] == account_type), None
    )
    return render_template("balance.html", member=member, savings=savings, now=date.today().isoformat())


@app.route("/member/<member_id>/open-subaccount", methods=["GET", "POST"])
def open_subaccount(member_id):
    member = MEMBERS.get(member_id)
    if not member:
        return redirect(url_for("not_found", member_id=member_id))

    if member_id in PERMISSION_DENIED_MEMBERS:
        return render_template("permission_denied.html", member_id=member_id)

    error = None
    if request.method == "POST":
        account_type = request.form.get("account_type", "").strip()
        initial_deposit = request.form.get("initial_deposit", "").strip()
        purpose = request.form.get("purpose", "").strip()

        if not account_type or account_type not in SUB_ACCOUNT_TYPES:
            error = "Please select a valid account type."
        elif not initial_deposit:
            error = "Please enter an initial deposit amount."
        else:
            try:
                deposit_val = float(initial_deposit.replace(",", ""))
                if deposit_val < 25:
                    error = "Minimum initial deposit is $25.00."
            except ValueError:
                error = "Initial deposit must be a numeric amount."

        if not error:
            session["pending_subaccount"] = {
                "member_id": member_id,
                "account_type": account_type,
                "initial_deposit": initial_deposit,
                "purpose": purpose,
            }
            return redirect(url_for("confirm_subaccount", member_id=member_id))

    return render_template(
        "open_subaccount.html",
        member=member,
        account_types=SUB_ACCOUNT_TYPES,
        error=error,
    )


@app.route("/member/<member_id>/open-subaccount/confirm", methods=["GET", "POST"])
def confirm_subaccount(member_id):
    member = MEMBERS.get(member_id)
    if not member:
        return redirect(url_for("not_found", member_id=member_id))

    pending = session.get("pending_subaccount")
    if not pending or pending.get("member_id") != member_id:
        return redirect(url_for("open_subaccount", member_id=member_id))

    if request.method == "POST":
        # Simulate account creation success
        session.pop("pending_subaccount", None)
        new_account_number = "****" + str(9000 + len(MEMBERS[member_id]["accounts"]))
        new_account = {
            "type": pending["account_type"],
            "number": new_account_number,
            "balance": float(pending["initial_deposit"].replace(",", "")),
        }
        MEMBERS[member_id]["accounts"].append(new_account)
        return render_template(
            "subaccount_success.html",
            member=member,
            account=new_account,
        )

    return render_template("confirm_subaccount.html", member=member, pending=pending)


@app.route("/not-found")
def not_found():
    member_id = request.args.get("member_id", "")
    return render_template("not_found.html", member_id=member_id)


@app.route("/reports/slow-summary")
def slow_summary():
    """Simulates a slow back-end call — used to exercise bounded wait/retry."""
    time.sleep(4)
    return render_template("slow_summary.html")


@app.route("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
