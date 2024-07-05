import gradio as gr
import json

# Function to read JSONL file and parse entries
def read_jsonl_file(file_path):
    entries = []
    with open(file_path, 'r') as file:
        for line in file:
            entries.append(json.loads(line.strip()))
    return entries

# Load entries from JSONL file
jsonl_file_path = 'questions.jsonl'
entries = read_jsonl_file(jsonl_file_path)

# Function to display details based on selected question_id
def display_details(question_id):
    for entry in entries:
        if entry["question_id"] == question_id:
            turn_details = []
            for i in range(3):
                context = entry["contexts"][i] if i < len(entry["contexts"]) else ""
                if isinstance(context, list):
                    context = " ".join(context)
                
                turn_detail = {
                    "Query": entry["turns"][i] if i < len(entry["turns"]) else "",
                    "Context": context,
                    "Reference": entry["references"][i] if i < len(entry["references"]) else "",
                }
                if entry['aspect'] == "CR":
                    turn_detail["Evidence Text"] = entry["evidence_text"][i] if i < len(entry["evidence_text"]) else ""
                turn_details.append(turn_detail)
            return turn_details
    return [{"Query": "", "Context": "", "Reference": "", "Evidence Text": ""} for _ in range(3)]

# Function to get unique categories
def get_categories():
    categories = list(set(entry["category"] for entry in entries))
    return categories

# Function to get question_ids based on category and aspect
def get_question_ids(category, aspect):
    question_ids = [entry["question_id"] for entry in entries if entry["category"] == category and entry["aspect"] == aspect]
    return gr.update(choices=question_ids, value=None)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Interactive JSONL Viewer")
    
    aspect = gr.Dropdown(label="Select Aspect", choices=["CS", "CR", "TR"])
    category = gr.Dropdown(label="Select Category", choices=get_categories())
    question_id = gr.Dropdown(label="Select Question ID", choices=[], interactive=True)
    
    def update_question_ids(category, aspect):
        return get_question_ids(category, aspect)
    
    category.change(fn=update_question_ids, inputs=[category, aspect], outputs=question_id)
    aspect.change(fn=update_question_ids, inputs=[category, aspect], outputs=question_id)
    
    display_button = gr.Button("Display Details")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Turn 1")
            query1 = gr.Textbox(label="Query", interactive=False)
            context1 = gr.Textbox(label="Context", interactive=False)
            reference1 = gr.Textbox(label="Reference", interactive=False)
            evidence_text1 = gr.Textbox(label="Evidence Text", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Turn 2")
            query2 = gr.Textbox(label="Query", interactive=False)
            context2 = gr.Textbox(label="Context", interactive=False)
            reference2 = gr.Textbox(label="Reference", interactive=False)
            evidence_text2 = gr.Textbox(label="Evidence Text", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Turn 3")
            query3 = gr.Textbox(label="Query", interactive=False)
            context3 = gr.Textbox(label="Context", interactive=False)
            reference3 = gr.Textbox(label="Reference", interactive=False)
            evidence_text3 = gr.Textbox(label="Evidence Text", interactive=False)
    
    def update_ui(question_id):
        details = display_details(question_id)
        return (
            details[0]["Query"], details[0]["Context"], details[0]["Reference"], details[0].get("Evidence Text", ""),
            details[1]["Query"], details[1]["Context"], details[1]["Reference"], details[1].get("Evidence Text", ""),
            details[2]["Query"], details[2]["Context"], details[2]["Reference"], details[2].get("Evidence Text", "")
        )
    
    display_button.click(
        fn=update_ui, 
        inputs=question_id, 
        outputs=[query1, context1, reference1, evidence_text1, query2, context2, reference2, evidence_text2, query3, context3, reference3, evidence_text3]
    )

demo.launch(server_port=7777, server_name="0.0.0.0")