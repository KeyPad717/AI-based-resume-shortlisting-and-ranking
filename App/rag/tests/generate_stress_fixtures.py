"""Generate synthetic PDF fixtures for the Generalization Stress Test (Task A).

These fixtures are deliberately different from the 7 real sample resumes to
stress the existing chunker (ResumeIngestor) WITHOUT modifying it:
  Fixture A: two-column layout
  Fixture B: non-(bullet) list style (numbers + dashes)
  Fixture C: out-of-vocabulary section headers (Publications, Awards, etc.)
"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIXTURES, exist_ok=True)


def _text_width(c, s):
    return c.stringWidth(s, "Helvetica", 10)


def fixture_two_column(path):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "ALEX RIVERA")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 26 * mm, "Senior Backend Engineer  |  alex.rivera@mail.com")

    left_x = 20 * mm
    right_x = 110 * mm
    y = height - 38 * mm

    # LEFT COLUMN: skills
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_x, y, "SKILLS")
    left_items = [
        "Python, Go, Rust",
        "Kubernetes, Docker",
        "PostgreSQL, Redis",
        "gRPC, REST, GraphQL",
        "AWS, Terraform",
        "System Design",
        "Distributed Systems",
    ]
    c.setFont("Helvetica", 9)
    y_ = y - 8 * mm
    for item in left_items:
        c.drawString(left_x, y_, item)
        y_ -= 7 * mm

    # RIGHT COLUMN: experience (reading order interleaves with left col)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(right_x, y, "EXPERIENCE")
    blurb = (
        "Lead a team of 4 building the checkout platform. "
        "Reduced p95 latency by 40% via caching and query tuning. "
        "Cut k8s deployment time from 15 to 3 minutes."
    )
    c.setFont("Helvetica", 9)
    words = blurb.split()
    line = ""
    y_ = y - 8 * mm
    for w in words:
        trial = (line + " " + w).strip()
        if _text_width(c, trial) > 75 * mm and line:
            c.drawString(right_x, y_, line)
            y_ -= 7 * mm
            line = w
        else:
            line = trial
    if line:
        c.drawString(right_x, y_, line)

    c.showPage()
    c.save()


def fixture_numbered_dashes(path):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "MIRA PATEL")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 26 * mm, "Data Scientist")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, height - 38 * mm, "EXPERIENCE")
    lines = [
        "01  Built a fraud-detection model in Python using gradient boosting",
        "02  Engineered feature pipelines with Apache Spark and Airflow",
        "03  Reduced fraud false positives by 25% across three product lines",
        "04  Deployed models to production via MLflow and Docker",
    ]
    c.setFont("Helvetica", 9)
    y = height - 46 * mm
    for line in lines:
        c.drawString(20 * mm, y, "- " + line)
        y -= 7 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y - 6 * mm, "OUT-OF-VOCAB HEADER BELOW")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y - 14 * mm, "Skills: pandas, numpy, scikit-learn, Spark, SQL")

    c.showPage()
    c.save()


def fixture_oov_headers(path):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "JORDAN KIM")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 26 * mm, "Sr. Full-Stack Engineer")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, height - 38 * mm, "PUBLICATIONS")
    pubs = [
        "R. Kim, 'A Survey of Edge Caching Strategies', IEEE 2023",
        "J. Kim et al., 'Stateful Microservices on Kubernetes', Workshop 2022",
    ]
    c.setFont("Helvetica", 9)
    y = height - 46 * mm
    for p in pubs:
        c.drawString(20 * mm, y, p)
        y -= 7 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y - 6 * mm, "AWARDS")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y - 14 * mm, "Best Paper Runner-Up, 2022; Dean's List 2019")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y - 30 * mm, "CERTIFICATIONS")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y - 38 * mm, "AWS Solutions Architect; CNCF CKAD")

    c.showPage()
    c.save()


def fixture_date_range_bullets(path):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "DANA OKAFOR")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 26 * mm, "Software Engineer")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, height - 38 * mm, "PROJECTS")
    lines = [
        "1. Built a load-balancing system Jan 2023 - Mar 2024 reducing downtime",
        "2. Led a team of three to ship the payments platform",
        "3. Wrote integration tests for the checkout flow",
    ]
    c.setFont("Helvetica", 9)
    y = height - 46 * mm
    for line in lines:
        c.drawString(20 * mm, y, line)
        y -= 11 * mm

    c.showPage()
    c.save()


fixture_two_column(os.path.join(FIXTURES, "fixture_two_column.pdf"))
fixture_numbered_dashes(os.path.join(FIXTURES, "fixture_numbered_dashes.pdf"))
fixture_oov_headers(os.path.join(FIXTURES, "fixture_oov_headers.pdf"))
fixture_date_range_bullets(os.path.join(FIXTURES, "fixture_date_range_bullets.pdf"))
print("Wrote 4 fixture PDFs to", FIXTURES)
