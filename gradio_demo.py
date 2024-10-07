import gradio as gr
from inference import inference


def send_to_model(source_video, prompt, neg_prompt, guidance_scale, video_length, old_qk):
    return inference(
        prompt=prompt, neg_prompt=neg_prompt, guidance_scale=guidance_scale, video_length=video_length, video_path=source_video, old_qk=old_qk
    )


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <h1 style="text-align: center; font-size: 32px; font-family: 'Times New Roman', Times, serif;">
                FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing
            </h1>
            <p style="text-align: center; font-size: 20px; font-family: 'Times New Roman', Times, serif;">
                <a style="text-align: center; display:inline-block"
                    href="https://flatten-video-editing.github.io/">
                    <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/paper-page-sm.svg#center"
                    alt="Paper Page">
                </a>
                <a style="text-align: center; display:inline-block" href="https://huggingface.co/spaces/sky24h/FLATTEN-unofficial?duplicate=true">
                    <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm.svg#center" alt="Duplicate Space">
                </a>
            </p>
            """
        )
        gr.Interface(
            fn=send_to_model,
            inputs=[
                gr.Video(value=None, label="source_video"),
                gr.Textbox(value="", label="prompt"),
                gr.Textbox(value="", label="neg_prompt"),
                gr.Slider(
                    value   = 15,
                    minimum = 10,
                    maximum = 30,
                    step    = 1,
                    label   = "guidance_scale",
                    info    = "The scale of the guidance field.",
                ),
                gr.Slider(
                    value   = 16,
                    minimum = 8,
                    maximum = 32,
                    step    = 2,
                    label   = "video_length",
                    info    = "The length of the video, must be less than 16 frames in the online demo to avoid timeout. However, you can run the model locally to process longer videos.",
                ),
                gr.Dropdown(value=0, choices=[0, 1], label="old_qk", info="Select 0 or 1."),
            ],
            outputs        = [gr.Video(label="output", autoplay=True)],
            allow_flagging = "never",
            description    = "This is an unofficial demo for the paper 'FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing'.",
        )
        demo.queue(max_size=10).launch()
