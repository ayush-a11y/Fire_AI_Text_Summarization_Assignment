from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize FastAPI app
app = FastAPI()

# Define the input structure for the summarization API
class SummarizationRequest(BaseModel):
    prompt: str

# Load your trained T5 summarization model (modify the path to your model if needed)
model_path = '/home/konsultera/konsultera_work/Learning_and_Practice/fire_ai_task/t5_model_code_with_inference_and_checkpoints/t5_samll_model_checkpoints/t5_samsum-20240923T062318Z-001/t5_samsum'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Device configuration (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the summarization endpoint
@app.post("/summarise")
async def summarise_text(request: SummarizationRequest):
    try:
        # Preprocess the input text
        inputs = tokenizer.encode("summarize: " + request.prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        # Generate the summary
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {"response": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with Uvicorn (for local development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

