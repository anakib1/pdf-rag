import gradio as gr
from dotenv import load_dotenv
from src.clients import AcademicClient

load_dotenv()
client = AcademicClient()


def perform_qa(query):
    return client.answer(query)


css = """
body {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown('Wisdom.AI'),
    gr.Image('misc/wisdom.jpg', height=600, width=400)
    with gr.Row():
        inp = gr.Textbox('Що б ви хотіли дізнатися у мудрого?')
        out = gr.Textbox('Мудрий каже...')

    btn = gr.Button('Спитати')
    btn.click(fn=perform_qa, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()
