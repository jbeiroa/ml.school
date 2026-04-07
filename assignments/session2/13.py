"""
Create a flow that starts by generating a list of dictionaries using an LLM. Each item of the list will represent a student, and each student will have a name and score. The flow should use a foreach loop to process each student on a separate branch. Each branch will transform the student's name to uppercase and increase the score by a fixed amount (e.g., add 10). In the join step, the flow will aggregate all of the scores and print both the updated dictionaries and the aggregate result.
"""

from metaflow import FlowSpec, step, card, environment
import openai
import os
from pydantic import BaseModel

class StudentModel(BaseModel):
    name: str
    score: int

class ResponseModel(BaseModel):
    output_parsed: list[StudentModel]


class OpenAIFlow(FlowSpec):
    @environment(
        vars={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
    )
    @step
    def start(self):
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        response = client.responses.parse(
            model="gpt-4o-mini",
            instructions="You create list of dictionaries with random names and scores. Output only the list of dictionaries with no decorators or explanations.",
            input="Generate a list of 10 dictionaries, each containing a 'student' key with a unique random name as the value, and a 'score' key with a random integer between 0 and 100 as the value.",
            text_format=ResponseModel)
        self.students = response.output_parsed.output_parsed
        print(self.students)

        self.next(self.process_students, foreach="students")

    @step
    def process_students(self):
        student = self.input
        self.capitalized_name = student.name.upper()
        self.score = student.score + 10
        print(f"Processed Student: {self.capitalized_name}, Score: {self.score}")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.students = [{"name": i.capitalized_name, "score": i.score} for i in inputs]
        print(self.students)
        self.total_score = sum(i.score for i in inputs)
        print(f"Total Score: {self.total_score}")
        self.next(self.end)


    @step
    def end(self):
        print("Updated Students:", self.students)

if __name__ == "__main__":
    OpenAIFlow()