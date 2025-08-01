Todo Application with Streamlit
Overview
This is a Streamlit-based Todo application that allows users to manage tasks through a web interface. The app supports creating, viewing, updating, and deleting todo items stored in a MongoDB database. It uses a multi-agent system powered by the agents library and integrates with the Gemini API for processing user requests intelligently.
Features

Create Todos: Add new todo items with a title and description.
View Todos: Display all todos in a tabular format with ID, title, description, and completion status.
Update Todos: Modify existing todos by ID, title, or bulk update all todos.
Delete Todos: Remove todos by ID, title, or delete all todos.
Conversation History: View the history of user interactions with the assistant.
Responsive UI: Clean and intuitive interface with Streamlit tabs for each operation.

Prerequisites

Python 3.8+
MongoDB database with a valid connection URI
Gemini API key for the language model
Required Python packages:
streamlit
pandas
pymongo
python-dotenv
agents (custom library for agent-based processing)



Installation

Clone the repository (if applicable) or save the provided app.py file.

Install dependencies:
pip install streamlit pandas pymongo python-dotenv agents


Set up environment variables:

Create a .env file in the project root.
Add the following variables:MONGODB_URI=your_mongodb_connection_string
GEMINI_API_KEY=your_gemini_api_key


Replace your_mongodb_connection_string with your MongoDB URI and your_gemini_api_key with your Gemini API key.


Run the application:
streamlit run app.py



Usage

Access the app:
Open your browser and navigate to the URL provided by Streamlit (typically http://localhost:8501).


Navigate the UI:
Create Todo: Enter a title and description, then click "Add Todo".
View Todos: Click "Refresh Todos" to see all todos in a table.
Update Todo: Provide an ID, current title, or select "Update All Todos", then specify new title, description, or completion status.
Delete Todo: Provide an ID, title, or select "Delete All Todos" to remove todos.
Conversation History: Expand the "Conversation History" section to view past interactions.


Interact with the app:
The app uses a triage agent to route requests to specialized agents for creating, reading, updating, or deleting todos.
Feedback messages (success or warning) are displayed for each action.



Project Structure

app.py: Main application file containing the Streamlit UI and backend logic.
.env: Environment file for storing MongoDB URI and Gemini API key (not included in version control).

Dependencies

Streamlit: For the web interface.
Pandas: For displaying todos in a tabular format.
PyMongo: For MongoDB database interactions.
python-dotenv: For loading environment variables.
agents: For agent-based request processing with the Gemini API.

Notes

Ensure MongoDB is running and accessible with the provided URI.
The Gemini API key must be valid and configured correctly.
The agents library is assumed to be a custom or proprietary library for agent-based processing. Ensure it is installed and compatible.
The app uses asyncio for handling asynchronous operations, so ensure your environment supports it.

Troubleshooting

MongoDB Connection Error: Verify the MONGODB_URI in the .env file and ensure the MongoDB server is running.
Gemini API Error: Check the GEMINI_API_KEY and ensure the API endpoint is accessible.
Missing Dependencies: Run pip install -r requirements.txt if you have a requirements.txt file, or install the listed packages manually.
UI Issues: Ensure you are using the latest version of Streamlit and a compatible browser.

License
This project is for personal use and provided as-is. Ensure you have the necessary permissions for any proprietary libraries (e.g., agents) used in the project.
