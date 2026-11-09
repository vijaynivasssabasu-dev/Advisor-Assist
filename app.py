"""Gradio application entry point."""

# app.py
import os, tempfile
import gradio as gr
from agent import run_advisor
from guardrails import redact_pii

def advise(question: str, csv_file) -> str:
    if not (question or "").strip():
        return "Please type a question."
    csv_path = None
    if csv_file is not None:                      # step 2: redact before anything sees it
        cleaned = redact_pii(open(csv_file.name).read())
        csv_path = os.path.join(tempfile.gettempdir(), "advisoriq_upload.csv")
        with open(csv_path, "w") as f:
            f.write(cleaned)
    return run_advisor(question, csv_path)        # steps 3-8

demo = gr.Interface(
    fn=advise,
    inputs=[gr.Textbox(label="Your question", lines=2),
            gr.File(label="Upload transactions (CSV) — optional", file_types=[".csv"])],
    outputs=gr.Markdown(label="AdvisorIQ says"),
    title="AdvisorIQ — Wealth-Management Copilot",
    description="Ask an investment question and (optionally) upload your transactions. "
                "Educational only, not financial advice.",
)

if __name__ == "__main__":
    # 0.0.0.0 = reachable from outside the container; PORT is what Render assigns.
    # The 7860 fallback keeps local runs working. Section 10 explains why this matters.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))