"""
Example
-------
python ask_gemini.py \\
    ../data/Transcripts/TNS-G-C1L1P-Mar02-E-Irma_q2_04-08.docx \\
    ../data/typing_gt/G-C1L1P-Mar02-E-Irma_q2_04-08.xlsx

Output
------
* ~0:00:25
    - Negotiation: Students 2 and 3 refine parameters for the 'range'
      function. Student 3 corrects input by revising "Space, 0" to
      "I mean, space, 15," adjusting understanding of syntax.
    - Typed by: Jacinto

* ~0:01:35
    - Negotiation: Students 2 and 3 clarify and confirm syntax for
      defining a range. Student 2 gives precise instructions
      ("Negative 10, coma, space and then 0"), Student 3 reinforces.
    - Typed by: Herminio

* ~0:02:45
    - Negotiation: Students 3 and 5 negotiate parameters for a new
      'range' input. They explore limits ("1000," "Negative 100 to 100"),
      correct syntax ("You're not putting negative"), and settle on values.
    - Typed by: Juan

* ~0:03:15
    - Negotiation: Students 3, 4, and 5 debate feasibility and syntax
      for a challenging 'range' input ("1 million" to "0," negatives).
      They clarify scale, debate capacity, and give conflicting instructions.
    - Typed by: Juan
"""

import os
import argparse
from dotenv import load_dotenv
from google import genai
import docx
import pandas as pd


# In case you have your Gemini API key in .env file.
# I have it in my .bashrc.
load_dotenv()


class HAQText:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it to a .env file or set the environment variable."
            )
        self.client = genai.Client(api_key=api_key)

    def simple_question(self, question: str)->None:
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=question,
        )
        print(response.text)

    def example_prompt(self, transcript_path: str, typing_gt_path: str):
        """An example prompt using transcript and typing ground truth."""

        # The personality that we want the gemini to take. A student, master, expert, newbie, etc.
        #     VJ: I am trying to imitate Hannah here.
        persona = (
            "You are an Assistant Professor (H.K.Lee) who joined a semester back in education department."
            "You You excel in identifying indicators for student learning in a collaborative learning environment."
        )

        # Instructions / Main task
        #     VJ: Here I am hoping that gemini will find instances where the students first went through
        #         a negotiaion and then made a meaningful/tangible contribution to the session.
        instructions = (
            "Identify meaningful negotiations followed by typing."
        )

        # Context
        context = (
            "Your output should have time stamps (approximation is okay) from the transcripts and the typing "
            "excel file where there is a meaningful negotiation followed by typing. Meaningful negotiaon is defined "
            "as 'Students clarify, refine, and rework CSM concepts through dynamic language use, drawing on relational interactions and contextual cues to shape meaning.'"
        )

        # Format
        format = (
            "Create a list. Each list should have time stamp and two sublists. The first one should contain why this is a meaningful negotiation "
            "and the second one should have the name pseudonyms of the students who worked after the meaningful negotiation."
        )

        # Audience
        audience = (
            "The audience are education researchers who you want to "
            "convince that using verbal and non-verbal data (like typing) "
            "will give better insight into collaborative learning process."
        )

        # Tone
        tone = (
            "Simple and direct."
        )

        # Data
        data = (
            "The transcript and typing data are provided below."
        )

        # Read transcript from docx
        doc = docx.Document(transcript_path)
        transcript_text = "\n".join(p.text for p in doc.paragraphs)

        # Read typing ground truth from xlsx (Human readable sheet)
        typing_df = pd.read_excel(typing_gt_path, sheet_name="Human readable")
        typing_text = typing_df.to_csv(index=False)

        # Everything together
        full_prompt = f"""
        ## Persona
        {persona}

        ## Instructions
        {instructions}

        ## Context
        {context}

        ## Format
        {format}

        ## Audience
        {audience}

        ## Tone
        {tone}

        ## Data
        {data}

        ### Transcript
        {transcript_text}

        ### Typing Data
        {typing_text}
        """

        # Send prompt to Gemini
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
        )
        print(response.text)


def parse_cmd_arguments():

    parser = argparse.ArgumentParser(description="A simple prompt for Human Activity Quantification (HAQ).")
    parser.add_argument("transcript_path", help="Full path to the transcript")
    parser.add_argument("typing_gt", help="Typing ground truth as e")

    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_cmd_arguments()
    haq_text = HAQText()
    haq_text.simple_question("Top 5 burrito places in Albuquerque? Only give me names.")
    haq_text.example_prompt(cmd_args.transcript_path, cmd_args.typing_gt)
