import torch
from langchain_huggingface import HuggingFacePipeline

class Model:
    """ Init any HuggingFace model """
    def __init__(self, model_id="HuggingFaceTB/SmolLM-135M"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 512,
                "top_k": 50,
                "temperature": 0.45,
            }
        )
        
        # self.llm_engine_hf = ChatHuggingFace(llm=llm)
        
    def ask(self, prompt):
        """ Query """
        return self.llm.invoke(prompt)