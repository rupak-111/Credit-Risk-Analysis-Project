// Builds a polished Word version of the credit risk report with charts embedded.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, LevelFormat, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign,
  PageNumber, PageBreak,
} = require("docx");

const OUT = path.join(__dirname, "outputs");
const CONTENT_W = 9360; // US Letter, 1" margins

// ---- palette ----------------------------------------------------------
const NAVY = "1F3864", BLUE = "2E75B6", LIGHT = "D6E4F0",
      GREY = "595959", HEADROW = "1F3864", ZEBRA = "EEF3FA";

// ---- helpers ----------------------------------------------------------
const img = (file, w, h) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 60 },
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync(path.join(OUT, file)),
    transformation: { width: w, height: h },
    altText: { title: file, description: file, name: file },
  })],
});

const caption = (txt) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: txt, italics: true, size: 18, color: GREY })],
});

const h1 = (txt) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(txt)] });
const h2 = (txt) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(txt)] });
const body = (txt) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun(txt)] });

const bullet = (runs) => new Paragraph({
  numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(runs) ? runs : [new TextRun(runs)],
});
const numbered = (runs) => new Paragraph({
  numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(runs) ? runs : [new TextRun(runs)],
});

// ---- table builder ----------------------------------------------------
function makeTable(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerCells = headers.map((t, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: HEADROW, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: t, bold: true, color: "FFFFFF", size: 19 })],
    })],
  }));
  const dataRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: { fill: ri % 2 ? ZEBRA : "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 50, bottom: 50, left: 110, right: 110 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: [new TextRun({ text: String(c), size: 19,
          bold: i === 0 && ri >= 0 ? false : false })],
      })],
    })),
  }));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headerCells }), ...dataRows],
  });
}

// ---- pull live numbers from results_summary.json ----------------------
const S = JSON.parse(fs.readFileSync(path.join(OUT, "results_summary.json")));

// =======================================================================
const children = [];

// ----- Cover -----------------------------------------------------------
children.push(
  new Paragraph({ spacing: { before: 2400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "PACIFIC CREST BANK", bold: true, size: 26, color: BLUE, characterSpacing: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "HONG KONG  ·  GROUP RISK & ANALYTICS", size: 18, color: GREY, characterSpacing: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 0 },
    border: { top: { style: BorderStyle.SINGLE, size: 12, color: NAVY, space: 1 } },
    children: [new TextRun({ text: "", size: 2 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 120 },
    children: [new TextRun({ text: "Credit Risk Assessment", bold: true, size: 56, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: "Retail & SME Lending Portfolio", size: 32, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: NAVY, space: 1 } },
    children: [new TextRun({ text: "", size: 2 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200 },
    children: [new TextRun({ text: "Prepared by Group Risk & Analytics", size: 22, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
    children: [new TextRun({ text: "June 2026  ·  Confidential", size: 20, color: GREY })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ----- TOC -------------------------------------------------------------
children.push(
  h1("Contents"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ----- Executive Summary ----------------------------------------------
children.push(h1("Executive Summary"));
children.push(body(`This report presents an end-to-end credit risk assessment of a 12,000-facility sample of Pacific Crest Bank's Hong Kong retail and SME lending book. The portfolio carries a 12-month default rate of ${(S.default_rate*100).toFixed(1)}%, concentrated in small-business facilities and unsecured revolving credit.`));
children.push(body(`We built and benchmarked three predictive models and a regulatory-style credit scorecard. The governed WOE scorecard achieves a Gini of ${S.scorecard_gini.toFixed(2)} (AUC ${S.scorecard_auc.toFixed(3)}, KS ${S.scorecard_ks.toFixed(2)}) — matching the black-box challengers while remaining fully transparent and auditable, the right trade-off for a regulated origination decision under HKMA model-risk expectations.`));
children.push(body(`The scorecard produces well-calibrated probabilities of default and rank-orders cleanly into five risk grades, with realised default rates ranging from 1.6% (Grade A) to 46.6% (Grade E). We recommend a hard 50% DSR cap, adoption of the five-grade scorecard for origination, risk-based pricing, and portfolio steering toward the secured book.`));

// KPI strip as a table
children.push(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: "Headline metrics", bold: true, color: NAVY, size: 22 })] }));
children.push(makeTable(
  ["Portfolio default rate", "Scorecard AUC", "Scorecard Gini", "Scorecard KS"],
  [[`${(S.default_rate*100).toFixed(1)}%`, S.scorecard_auc.toFixed(3), S.scorecard_gini.toFixed(2), S.scorecard_ks.toFixed(2)]],
  [2340, 2340, 2340, 2340],
));

// ----- 1. Portfolio Profile -------------------------------------------
children.push(h1("1. Portfolio Profile"));
children.push(makeTable(
  ["Metric", "Value"],
  [
    ["Facilities analysed", "12,000"],
    ["12-month default rate (90+ DPD)", `${(S.default_rate*100).toFixed(1)}%`],
    ["Products", "Personal Loan, Credit Card, Mortgage, SME Facility"],
    ["Currency", "HKD"],
    ["Default definition", "90+ days past due within 12 months"],
  ],
  [3360, 6000],
));
children.push(h2("Default rate by product"));
children.push(makeTable(
  ["Product", "Default rate"],
  [["SME Facility", "26.5%"], ["Credit Card", "14.7%"], ["Personal Loan", "11.6%"], ["Mortgage", "8.5%"]],
  [6360, 3000],
));
children.push(body("Small-business facilities and unsecured revolving credit run materially hotter than the secured mortgage book, consistent with their risk profile and the HKMA's LTV discipline on residential lending."));
children.push(img("01_target_and_product.png", 600, 213));
children.push(caption("Figure 1. Target distribution and default rate by product."));
children.push(img("02_segment_default_rates.png", 600, 163));
children.push(caption("Figure 2. Default rate by employment sector, residential status and district."));

// ----- 2. Key Risk Drivers --------------------------------------------
children.push(h1("2. Key Risk Drivers"));
children.push(body("Drivers were identified consistently across three independent lenses — Information Value (univariate), machine-learning feature importance, and scorecard points:"));
children.push(makeTable(
  ["Rank", "Driver", "Information Value", "Strength"],
  [
    ["1", "TU credit score", "1.24", "Very strong"],
    ["2", "Recent delinquencies (24m)", "0.65", "Very strong"],
    ["3", "Any prior delinquency", "0.42", "Strong"],
    ["4", "DSR headroom vs 50% guide", "0.38", "Strong"],
    ["5", "Debt Servicing Ratio (DSR)", "0.38", "Strong"],
    ["6", "Product type", "0.16", "Medium"],
  ],
  [900, 4060, 2400, 2000],
));
children.push(body("Bureau behaviour and affordability dominate. The TU score and recent delinquency history are the strongest signals, followed closely by the DSR family of affordability measures — direct empirical support for the HKMA's emphasis on debt-servicing capacity."));
children.push(img("03_risk_drivers.png", 600, 163));
children.push(caption("Figure 3. DSR and TU score by outcome; default rate by delinquency count."));
children.push(img("05_information_value.png", 510, 337));
children.push(caption("Figure 4. Information Value ranking of candidate features."));

// ----- 3. Model Performance -------------------------------------------
children.push(h1("3. Model Performance"));
children.push(body("Performance on a stratified 25% hold-out sample. All preprocessing is fit on the training split only to prevent leakage."));
children.push(makeTable(
  ["Model", "AUC", "Gini", "KS", "Avg. Precision"],
  [
    ["Logistic Regression", "0.842", "0.684", "0.516", "0.532"],
    ["Random Forest (challenger)", "0.835", "0.670", "0.502", "0.512"],
    ["Gradient Boosting (challenger)", "0.826", "0.652", "0.486", "0.499"],
    ["WOE Scorecard (deployed)", "0.826", "0.651", "0.510", "—"],
  ],
  [3760, 1400, 1400, 1400, 1400],
));
children.push(body("The non-linear challengers offer no meaningful uplift over the transparent logistic / scorecard approach. For a regulated lending decision the scorecard is the clear choice: interpretable, monotonic, governable, and statistically on par with the best challenger."));
children.push(img("06_roc_curves.png", 440, 376));
children.push(caption("Figure 5. ROC curves on the test set."));

// ----- 4. Risk Grades, Calibration & Expected Loss --------------------
children.push(h1("4. Risk Grades, Calibration & Expected Loss"));
children.push(body("The scorecard PD bands into five grades. Realised default rates track model PDs closely — evidence of good calibration — and expected-loss rates rank-order monotonically (EL = PD × LGD × EAD, LGD = 45%):"));
children.push(makeTable(
  ["Grade", "Facilities", "Actual Default", "Model PD", "EAD (HK$)", "Expected Loss (HK$)", "EL Rate"],
  [
    ["A (Prime)", "609", "1.6%", "2.1%", "391.2M", "3.4M", "0.87%"],
    ["B (Low)", "1,117", "5.7%", "5.0%", "723.9M", "16.2M", "2.24%"],
    ["C (Medium)", "477", "10.1%", "10.8%", "318.6M", "16.0M", "5.03%"],
    ["D (High)", "359", "22.8%", "21.4%", "299.6M", "29.2M", "9.75%"],
    ["E (Decline)", "438", "46.6%", "49.5%", "379.7M", "82.7M", "21.77%"],
  ],
  [1500, 1260, 1500, 1140, 1320, 1640, 1000],
));
children.push(body("Grade E holds roughly 15% of facilities but a disproportionate share of expected loss (HK$82.7M, about 54% of total modelled EL). The EL-rate spread (0.87% to 21.77%) is the basis for both risk-based pricing and IFRS 9 provisioning."));
children.push(img("09_risk_grades.png", 600, 203));
children.push(caption("Figure 6. Actual default rate and total expected loss by risk grade."));
children.push(img("08_score_distribution.png", 540, 265));
children.push(caption("Figure 7. Scorecard distribution — performing vs default."));

// ----- 5. Illustrative Origination Policy -----------------------------
children.push(h1("5. Illustrative Origination Policy"));
children.push(body("A policy that auto-declines Grade E yields, on the hold-out sample:"));
children.push(makeTable(
  ["Metric", "Value"],
  [["Approval rate", `${(S.policy_E_approval_rate*100).toFixed(1)}%`],
   ["Bad-capture (defaults avoided)", `${(S.policy_E_bad_capture*100).toFixed(1)}%`]],
  [6360, 3000],
));
children.push(body("Declining the worst ~15% of applications removes roughly half of all future defaults. Grade D (next ~12%) should route to manual underwriting rather than auto-approval, capturing further loss with human oversight."));
children.push(img("10_decision_matrix.png", 360, 311));
children.push(caption("Figure 8. Decision matrix at a Grade-E decline cut-off."));

// ----- 6. Recommendations ---------------------------------------------
children.push(h1("6. Recommendations"));
children.push(h2("Lending policy"));
children.push(numbered("Hard 50% DSR cap with a stressed-rate overlay; second-line sign-off for 50–65% exceptions; auto-decline above 65%."));
children.push(numbered("Adopt the five-grade scorecard (A–E) for origination — auto-decline Grade E, manual underwrite Grade D, auto-approve A–C within policy."));
children.push(numbered("Risk-based pricing aligned to grade-level EL rates so margin covers provisioning, especially on the SME and card books."));
children.push(numbered("Portfolio steering — sub-limits on high-EL unsecured / SME segments; grow the secured mortgage book where EL rates are lowest."));
children.push(h2("Provisioning & capital"));
children.push(numbered("Use grade-level expected loss as the input to IFRS 9 ECL staging and management overlays."));
children.push(h2("Model governance (HKMA-aligned)"));
children.push(numbered("Monitor stability with PSI on out-of-time samples; recalibrate at least annually."));
children.push(numbered("Add macroeconomic overlays (unemployment, HIBOR, residential price index) for forward-looking ECL."));
children.push(numbered("Conduct fair-lending / disparate-impact testing before deployment."));
children.push(numbered("Stand up monitoring: monthly score distribution, override rates, early-arrears tracking."));

// ----- 7. Methodology --------------------------------------------------
children.push(h1("7. Methodology & Reproducibility"));
children.push(bullet([new TextRun({ text: "Data: ", bold: true }), new TextRun("synthetic HKD retail credit portfolio generated with an economically grounded data-generating process (leverage, bureau history, affordability, product mix). No real customer data.")]));
children.push(bullet([new TextRun({ text: "Pipeline: ", bold: true }), new TextRun("loading → QA & cleaning → EDA → feature engineering → WOE/IV screening → ML benchmarking → WOE scorecard → calibration, risk grading & expected loss.")]));
children.push(bullet([new TextRun({ text: "Leakage control: ", bold: true }), new TextRun("all imputation, scaling and WOE binning are learned on the training split only and applied to the hold-out.")]));
children.push(bullet([new TextRun({ text: "Full analysis: ", bold: true }), new TextRun("HK_Credit_Risk_Analysis.ipynb (executed, with inline charts). All figures and tables are saved under outputs/.")]));

children.push(new Paragraph({ spacing: { before: 360 },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 4 } },
  children: [new TextRun({ text: "Prepared by Group Risk & Analytics, Pacific Crest Bank (Hong Kong). Synthetic data for demonstration of credit-risk methodology only; not a real customer portfolio and not investment or lending advice.", italics: true, size: 16, color: GREY })] }));

// =======================================================================
const doc = new Document({
  creator: "Group Risk & Analytics",
  title: "Credit Risk Assessment — Pacific Crest Bank (HK)",
  styles: {
    default: { document: { run: { font: "Arial", size: 21, color: "262626" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: NAVY, font: "Arial" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: BLUE, font: "Arial" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "D9D9D9", space: 2 } },
      children: [new TextRun({ text: "Pacific Crest Bank (HK) — Credit Risk Assessment", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { top: { style: BorderStyle.SINGLE, size: 4, color: "D9D9D9", space: 2 } },
      children: [new TextRun({ text: "Confidential   ·   Page ", size: 16, color: GREY }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY }),
        new TextRun({ text: " of ", size: 16, color: GREY }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Credit_Risk_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out, `(${(buf.length/1024).toFixed(0)} KB)`);
});
