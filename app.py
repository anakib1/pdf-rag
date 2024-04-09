import gradio as gr
from dotenv import load_dotenv
from src.clients import AcademicClient

load_dotenv()
client = AcademicClient()


def perform_qa(query: str, options: str) -> str:
    return client.answer(query, options.split('\n'))


css = """
body {
    image-align: center;
    display:block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown('# Wisdom.AI'),
    gr.Image('misc/wisdom.jpg', height=600, width=400)
    with gr.Row():
        inp = gr.Textbox('Що б ви хотіли дізнатися у мудрого?', label='Питання', min_width=400)
        out = gr.Textbox('Мудрий каже...', label='Відповідь', min_width=400)

    options = gr.Textbox(label='Варіанти відповіді:', min_width=800)

    btn = gr.Button('Спитати')
    btn.click(fn=perform_qa, inputs=[inp, options], outputs=out)

if __name__ == "__main__":
    demo.launch()
