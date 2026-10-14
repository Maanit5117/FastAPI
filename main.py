import string
from pydantic import BaseModel, Field, computed_field
from typing import Annotated,Literal, Optional
from fastapi import FastAPI , Path , HTTPException , Query 
from fastapi.responses import JSONResponse
import json

app = FastAPI()

class Patient(BaseModel):
    id: Annotated[str,Field(..., description = "Unique ID of the patient", example = "P001")]
    name: Annotated[str,Field(..., description = "Name of the patient", example = "John Doe")]
    city: Annotated[str,Field(..., description = "City of the patient", example = "New York")]
    age: Annotated[int,Field(...,gt =0 , lt=120, description = "Age of the patient", example = 30)]
    gender: Annotated[Literal["Male", "Female", "Other"],Field(..., description = "Gender of the patient", example = "Male")]
    height: Annotated[float,Field(...,gt =0, description = "Height of the patient in cm", example = 175.5)]
    weight: Annotated[float,Field(...,gt =0, description = "Weight of the patient in kg", example = 70.0)]


    @computed_field
    @property
    def BMI(self) -> float:
        bmi = round(self.weight / ((self.height / 100) ** 2), 2)
        return bmi

    @computed_field
    @property
    def verdict(self) -> str:
        bmi = self.BMI
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 24.9:
            return "Normal weight"
        elif 25 <= bmi < 29.9:
            return "Overweight"
        else:
            return "Obesity"



class PatientUpdate(BaseModel):
    name: Annotated[Optional[str], Field(default=None)]
    city: Annotated[Optional[str], Field(default=None)]
    age: Annotated[Optional[int], Field(default=None, gt=0)]
    gender: Annotated[Optional[Literal['male', 'female']], Field(default=None)]
    height: Annotated[Optional[float], Field(default=None, gt=0)]
    weight: Annotated[Optional[float], Field(default=None, gt=0)]


def load_data():
    with open("patients.json", "r") as f:
        data = json.load(f)
    return data

def save_data(data):
    with open("patients.json", "w") as f:
        json.dump(data, f)



@app.get("/")
def Hello():
    return {"message": "Hello, Patient management system API"}


@app.get("/about")
def About():
    return {"message": "Fully functional API to manage your patient records."}


@app.get("/view")
def View():
    data = load_data()
    return data


@app.get("/patient/{patient_id}")
def view_patient(patient_id: str = Path(..., description = "The ID of patient in the DB ", example = "P001")):
    #load all the patients
    data = load_data()

    if patient_id in data:
        return data[patient_id]
    raise HTTPException(status_code=404, detail=f"Patient with ID {patient_id} not found.")


@app.get("/sort")
def sort_patients(sort_by: str = Query(..., description = "Sort on the basis of height, weight or BMI")
                  , order : str = Query("asc", description = "Order of sorting, can be asc or desc")):
    valid_fields = ["height", "weight", "BMI"]

    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail=f"Invalid sort field. Must be one of {valid_fields}")

    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail=f"Invalid order. Must be one of ['asc', 'desc'].")

    data = load_data()

    sort_order = True if order == "desc" else False

    sorted_data = sorted(data.values(), key=lambda x: x.get(sort_by,0), reverse = sort_order)

    return sorted_data


@app.post("/create")
def create_patient(patient: Patient):
    data = load_data()

    if patient.id in data:
        raise HTTPException(status_code=400, detail=f"Patient with ID {patient.id} already exists.")

    data[patient.id] = patient.model_dump(exclude=['id']) 

    save_data(data)

    return JSONResponse(status_code=201, content={"message": f"Patient with ID {patient.id} created successfully."})



@app.put('/edit/{patient_id}')
def update_patient(patient_id: str, patient_update: PatientUpdate):

    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    
    existing_patient_info = data[patient_id]

    updated_patient_info = patient_update.model_dump(exclude_unset=True)

    for key, value in updated_patient_info.items():
        existing_patient_info[key] = value

    #existing_patient_info -> pydantic object -> updated bmi + verdict
    existing_patient_info['id'] = patient_id
    patient_pydandic_obj = Patient(**existing_patient_info)
    #-> pydantic object -> dict
    existing_patient_info = patient_pydandic_obj.model_dump(exclude='id')

    # add this dict to data
    data[patient_id] = existing_patient_info

    # save data
    save_data(data)

    return JSONResponse(status_code=200, content={'message':'patient updated'})



@app.delete('/delete/{patient_id}')
def delete_patient(patient_id: str):

    # load data
    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    
    del data[patient_id]

    save_data(data)

    return JSONResponse(status_code=200, content={'message':'patient deleted'})