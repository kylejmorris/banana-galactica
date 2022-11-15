import torch
import galai as gal

model = gal.load_model("standard")
model.generate("Scaled dot product attention:\n\n\\[")

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
   
    device = 0 if torch.cuda.is_available() else -1
    model = gal.load_model("standard")
 

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model.generate(prompt)

    # Return the results as a dictionary
    return result
