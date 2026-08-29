"""Generate a second job-description PDF for Generalization Stress Test Task C.

Only one JD (JD/JD.pdf, NetApp) exists in the repo, so this constructs a second,
unrelated JD (a different role/company) to exercise the existing extraction path.

This is a TEST FIXTURE only. Phase 3 code (App/rag/*) contains zero JD-specific
logic; the point of Task C is to show a second JD extracts cleanly through the
existing pipeline extract_text() helper and that nothing in Phase 3 branches on
JD content.
"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

JD2_TEXT = [
    "Senior Platform Engineer - CloudInfra Labs",
    "About the role",
    "We are looking for a Senior Platform Engineer to own the developer platform "
    "and cloud infrastructure for CloudInfra Labs, a mid-sized SaaS company.",
    "Responsibilities",
    "Design, build and operate Kubernetes-based platform services on AWS.",
    "Automate infrastructure with Terraform and CI/CD pipelines.",
    "Champion reliability, observability and cost efficiency across engineering.",
    "Collaborate with product teams to ship developer tooling.",
    "Requirements",
    "5+ years in platform engineering or SRE roles.",
    "Strong Kubernetes, Docker and Linux administration skills.",
    "Experience with Terraform, Ansible and cloud networking on AWS.",
    "Scripting proficiency in Python or Go.",
    "Nice to have",
    "Experience with GitOps (Argo CD), service mesh (Istio), Prometheus.",
]


def main():
    c = canvas.Canvas(os.path.join(FIXTURES, "JD2_cloudinfra.pdf"), pagesize=A4)
    width, height = A4
    y = height - 20 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, JD2_TEXT[0])
    y -= 12 * mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y, JD2_TEXT[1])
    y -= 7 * mm
    c.setFont("Helvetica", 9)
    for line in JD2_TEXT[2:]:
        c.drawString(20 * mm, y, line)
        y -= 7 * mm
        if y < 25 * mm:
            c.showPage()
            y = height - 20 * mm
    c.showPage()
    c.save()
    print("Wrote", os.path.join(FIXTURES, "JD2_cloudinfra.pdf"))


if __name__ == "__main__":
    main()
