import {
	buildPptx,
	createDeck,
} from "/data/skills/pptx/scripts/pptx_builder_runtime.mjs"
import { applyRecipeDeckPlan } from "/data/skills/pptx/scripts/slide_recipes.mjs"

// Every number in this deck is measured from the working prototype in this
// repository. The corpus is 13 generated documents: 10 with a real PDF text
// layer and 3 deliberately degraded scans.
const IMG = "/data/sih26018/deck/img"

const deck = createDeck({ width: 1280, height: 720 })

const slidePlan = [
	{
		recipeId: "cover-editorial",
		content: {
			eyebrow: "SIH 2026 \u00b7 PS 26018",
			title: "Intelligent Land Record Digitization",
			subtitle:
				"Reads Indian land records, validates them, scores its own confidence, and routes anything uncertain to a human.",
			section: "Department of Land Resources",
		},
	},
	{
		recipeId: "big-claim",
		content: {
			eyebrow: "The problem",
			claim: "A wrong khasra number is not a data error. It is a property dispute.",
			support:
				"Decades of khatauni and 7/12 extracts sit in tehsil offices as paper.",
		},
	},
	{
		recipeId: "evidence-panel",
		content: {
			eyebrow: "Why this is hard",
			title: "Reading characters is the easy part",
			claim: "The difficulty is knowing which field a value belongs to.",
			evidence: [
				["Bilingual rows", "Value follows the final separator"],
				["Regional units", "A bigha differs by state"],
				["Rival labels", "Two date fields collide"],
			],
		},
	},
	{
		recipeId: "process-steps",
		content: {
			eyebrow: "How it works",
			title: "Three stages, each one auditable",
			steps: [
				{ title: "Extract", body: "Text layer, then OCR, then degraded mode." },
				{ title: "Classify", body: "Map lines onto 17 land-record fields." },
				{ title: "Validate", body: "Rules, master data, then route." },
			],
		},
	},
	{
		recipeId: "image-left-story",
		content: {
			eyebrow: "Verification",
			title: "Page and reasoning, side by side",
			image: { src: `${IMG}/ws.png`, fit: "contain" },
			body:
				"Each field carries a confidence bar and a one-click confirm. Clicking a value makes it editable, and trust recalculates on save. This record scored 94.",
			source: "Prototype \u00b7 sample_01_khatauni_up_clean.pdf",
		},
	},
	{
		recipeId: "metric-grid",
		content: {
			eyebrow: "Measured, not estimated",
			title: "What it does today",
			metrics: [
				{ value: "13", label: "documents end to end" },
				{ value: "94%", label: "completeness, clean records" },
				{ value: "342 ms", label: "average per page" },
				{ value: "17", label: "fields classified" },
				{ value: "0", label: "installs required" },
				{ value: "100%", label: "actions audited" },
			],
		},
	},
	{
		recipeId: "chart-led",
		content: {
			eyebrow: "Routing",
			title: "The system sorts its own output",
			chart: {
				chartType: "column",
				categories: ["Auto-approved", "Needs review", "Blocked"],
				series: [{ name: "Documents", values: [2, 4, 7], unit: "documents" }],
				valueAxisTitle: "Documents",
				showValueAxis: true,
			},
			takeaway:
				"Seven blocked is the correct answer: three are degraded scans, the rest carry real defects.",
		},
	},
	{
		recipeId: "table-led",
		content: {
			eyebrow: "Validation",
			title: "Rules firing on the sample corpus",
			table: {
				rows: [
					["Rule", "Severity", "What it caught"],
					["REQUIRED_MISSING", "Error", "Mandatory field absent, 18 times"],
					["DUPLICATE_CONFLICT", "Error", "Same parcel, different owner"],
					["DATE_FUTURE", "Error", "Trust fell to 26"],
					["DISTRICT_UNKNOWN", "Warning", "Suggested 'Kanpur Nagar'"],
					["AREA_REGIONAL_UNIT", "Warning", "Bigha to 6,322.98 m\u00b2, flagged"],
					["OWNER_FATHER_SAME", "Error", "Same person in both fields"],
				],
			},
			takeaway:
				"Findings name the field and the rule, so a reviewer is told what to fix.",
		},
	},
	{
		recipeId: "image-left-story",
		content: {
			eyebrow: "Learning loop",
			title: "Corrections become improvements",
			image: { src: `${IMG}/learn.png`, fit: "contain" },
			body:
				"Confusions, aliases and confidence calibration are mined from human corrections. Rules activate only past a support threshold, so one typo never reshapes the model.",
			source: "Prototype \u00b7 Learning tab",
		},
	},
	{
		recipeId: "comparison",
		content: {
			eyebrow: "Governance",
			title: "Access control and audit are enforced",
			leftTitle: "Four roles, server-side",
			leftBody:
				"Operators upload and correct. Inspectors approve and reject. Officers retrain and export. The audit cell is read-only. An operator attempting approval gets a 403.",
			rightTitle: "Append-only trail",
			rightBody:
				"Ingestion, corrections with old and new value, approvals and rejections are written with actor and timestamp. Nothing can edit a past entry.",
		},
	},
	{
		recipeId: "image-left-story",
		content: {
			eyebrow: "Dashboard",
			title: "Reporting the department can act on",
			image: { src: `${IMG}/dash.png`, fit: "contain" },
			body:
				"Volume, pending review, trust, district progress, error frequency and per-field quality. Precision reads 0.0% rather than an invented number, because only one field is reviewed so far.",
			source: "Prototype \u00b7 Dashboard tab",
		},
	},
	{
		recipeId: "two-column",
		content: {
			eyebrow: "Engineering choices",
			title: "Two decisions that make the demo survivable",
			left:
				"Zero install. Python standard library only, plus plain HTML and JavaScript. The biggest demo risk is a venue with no internet and a failing pip install.",
			right:
				"Honest degraded mode. With no OCR engine, a scan is blocked with trust 0 and enters the queue as work, never as a silent success.",
		},
	},
	{
		recipeId: "timeline-clean",
		content: {
			eyebrow: "Roadmap",
			title: "From prototype to district pilot",
			events: [
				{ label: "Now", caption: "Working prototype" },
				{ label: "Stage 2", caption: "Indic HTR" },
				{ label: "Stage 3", caption: "Full LGD data" },
				{ label: "Stage 4", caption: "DILRMP connectors" },
				{ label: "Stage 5", caption: "District pilot" },
			],
		},
	},
	{
		recipeId: "closing-takeaways",
		content: {
			eyebrow: "Summary",
			title: "Three things to remember",
			takeaways: [
				"A running system, not a mockup.",
				"Confidence is explainable, not a bare score.",
				"It refuses to fake success.",
			],
		},
	},
	{
		recipeId: "source-summary",
		content: {
			eyebrow: "Notes",
			title: "Scope and limitations",
			sources: [
				"PS 26018, Dept of Land Resources",
				"No dataset supplied; corpus generated",
				"OCR tier pluggable, not yet benchmarked",
				"Handwriting needs an Indic HTR model",
				"Bigha conversion uses UP values",
				"Master data covers eight states",
			],
		},
	},
]

applyRecipeDeckPlan(deck, slidePlan)

await buildPptx(deck, {
	scenePath: "/data/sih26018/deck/build/sih26018.scene.json",
	outputPath: "/data/sih26018/deck/SIH26018_Land_Record_Digitization.pptx",
	reportPath: "/data/sih26018/deck/build/sih26018.build-report.json",
	previewDir: "/data/sih26018/deck/build/preview",
	layoutDir: "/data/sih26018/deck/build/layout",
})
