from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5216"}})

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Initialize the ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

def extract_text_by_keywords(pdf_path, keywords):
    extracted_text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text = page.get_text("text")
            if any(keyword.lower() in text.lower() for keyword in keywords):
                extracted_text += text + "\n"
    return extracted_text.strip()

def summarize_text(text, max_length=130, min_length=30):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('pdf_files')
        
        if not files or len(files) != 3:
            return jsonify({"error": "Please upload exactly 3 PDF files."}), 400
        
        summaries = ["", "", "", ""]
        llm_response = ""
        
        try:
            keywords = ["Conclusions", "DISCUSSION AND CONCLUSION",
                       "LIMITATION AND RECOMMENDATION", "CONCLUSION AND DISCUSSION"]
            
            for i, file in enumerate(files):
                if file:
                    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(pdf_path)
                    extracted_text = extract_text_by_keywords(pdf_path, keywords)
                    if extracted_text:
                        summaries[i] = summarize_text(extracted_text)
            
            description_1 = summaries[0]
            description_2 = summaries[1]
            description_3 = summaries[2]
            
            prompt_extract = PromptTemplate(
                input_variables=["description_1", "description_2", "description_3"],
                template="Give me the best description about the research gaps using given details and give me only one paragraph: \"{description_1}, {description_2}, {description_3}\""
            )
            
            formatted_prompt = prompt_extract.format(
                description_1=description_1,
                description_2=description_2,
                description_3=description_3,
            )
            
            response = llm.invoke(formatted_prompt)
            llm_response = response.content
            
            return jsonify({
                "summary1": summaries[0],
                "summary2": summaries[1],
                "summary3": summaries[2],
                "llm_response": llm_response
            })
            
        except Exception as e:
            print(f"Error processing files: {str(e)}")
            return jsonify({"error": "Error processing files"}), 500
    
    return jsonify({"message": "API is running"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
