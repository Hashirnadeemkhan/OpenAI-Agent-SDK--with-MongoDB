import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,set_tracing_disabled
from pydantic import BaseModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import List, Optional,Union
from bson import ObjectId
import asyncio
from pymongo.errors import PyMongoError


# Load environment variables from .env file
load_dotenv()

uri = os.getenv("MONGODB_URI")
if not uri:
    raise ValueError("MONGODB_URI not set in environment variables")
client:MongoClient = MongoClient(uri, server_api=ServerApi('1'))
collection = client['todo_db']['todos']

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
# Get Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize OpenAI provider with Gemini API settings
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# Configure the language model
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)

# Pydantic models for structured output
class TodoItem(BaseModel):
    """A todo item with an ID, title, description, and completion status."""
    id: str
    title: str
    description: str
    completed: bool

class TodoList(BaseModel):
    todos: List[TodoItem]
    

# MongoDB CRUD operations as tools
@function_tool
async def create_todo(title: str, description: str) -> str:
    """Create a new todo item with given title and description."""
    todo = {
        "title": title,
        "description": description,
        "completed": False
    }
    result = collection.insert_one(todo)
    return f"Todo created with ID: {str(result.inserted_id)}"

@function_tool
async def read_todos() -> TodoList:
    try:
        todos = collection.find()
        todo_list = [
            TodoItem(id=str(todo["_id"]), title=todo["title"], description=todo["description"], completed=todo["completed"])
            for todo in todos
        ]
        return TodoList(todos=todo_list)
    except PyMongoError as e:
        raise Exception(f"Error reading todos: {str(e)}")


@function_tool
async def update_todo(
    id: str = "",
    new_title: str = "",
    description: str = "",
    completed: bool = None,
    match_title: str = "",
    mark_all: bool = False
) -> str:
    """
    Update todo item(s) with various fields.
    
    Args:
        id: ID of specific todo to update
        new_title: New title to set (empty means don't change)
        description: New description to set (empty means don't change)
        completed: Set completion status (None means don't change)
        match_title: Current title to match for update
        mark_all: Update all todos if True
        
    Returns:
        str: Confirmation message of the update
    """
    update_data = {}
    if new_title:
        update_data["title"] = new_title
    if description:
        update_data["description"] = description
    if completed is not None:
        update_data["completed"] = completed
    
    if not update_data:
        return "No update fields provided (new_title, description, or completed)"
    
    if mark_all:
        result = collection.update_many({}, {"$set": update_data})
        return f"Updated {result.modified_count} todos."
    
    if id:
        result = collection.update_one({"_id": ObjectId(id)}, {"$set": update_data})
        return f"Updated todo with ID {id}."
    
    if match_title:
        result = collection.update_many(
            {"title": {"$regex": match_title, "$options": "i"}},
            {"$set": update_data}
        )
        return f"Updated {result.modified_count} todos matching '{match_title}'."
    
    return "Please provide an ID, match_title, or set mark_all=True."


@function_tool
async def delete_todo(
    id: str = "",
    title: str = "",
    delete_all: bool = False
) -> str:
    """
    Delete a todo item by ID, title, or delete all todos.

    Arguments:
    - id: The ID of the todo item to delete.
    - title: The title of the todo item to delete.
    - delete_all: Set to True to delete all todos.
    """
    if delete_all:
        result = collection.delete_many({})
        return f"Deleted {result.deleted_count} todo item(s)."

    if id:
        result = collection.delete_one({"_id": ObjectId(id)})
        return f"Deleted {result.deleted_count} todo item(s) with ID {id}"

    if title:
        result = collection.delete_many({"title": {"$regex": title, "$options": "i"}})
        return f"Deleted {result.deleted_count} todo item(s) matching title '{title}'"

    return "Please provide an ID, title, or set delete_all=True."


# Agent definitions with optimized prompts
create_agent = Agent(
    name="CreateAgent",
    instructions="""
You are a helpful assistant who creates todo items using the `create_todo` tool.

Your goals:
- If the user asks to add one or more todo items, even indirectly (e.g., "make a todo", "add 5 sample todos", or "add something to my list"), you should create them.
- If the user asks you to create todos on your own (e.g. "add 5 example todos", "generate todos", "create some tasks"), then come up with useful, realistic todos on your own — like daily chores, health goals, work/study tasks.
- For each todo you create, use the `create_todo` tool and print the returned ID.
- Be friendly, fast, and don’t ask too many questions. Assume the user wants results quickly.

Examples of valid user requests:
- "Create 5 example todos"
- "Add a few new tasks like clean room, wash dishes"
- "Make some study reminders"

""",
    tools=[create_todo],
    model=model
)


read_agent = Agent(
    name="ReadAgent",
    instructions="You are a specialized agent for retrieving todo items. Use the read_todos tool to fetch and display all todos in a clear, organized format. Summarize the list concisely.",
    tools=[read_todos],
    model=model
)

update_agent = Agent(
    name="UpdateAgent",
    instructions="""
You are a powerful todo editing assistant that can modify todos in any way.

When user requests to change a todo title:
1. Extract both the current title (to match) and new title
2. Use match_title parameter for the current title
3. Use new_title parameter for the new title
4. Example: For "change Review Work Tasks to buy a car", call:
   update_todo(match_title="Review Work Tasks", new_title="buy a car")

You can also:
- Update descriptions
- Change completion status
- Make bulk updates
""",
    tools=[update_todo],
    model=model
)

delete_agent = Agent(
    name="DeleteAgent",
    instructions="""
You are a helpful assistant for deleting todo items.

- Use the delete_todo tool.
- You can delete by ID, by title (or partial title), or delete all todos if the user requests.
- If the user says something like "delete all", use `delete_all=True`.
- Be smart — infer the user's intent and match the right todos.
""",
    tools=[delete_todo],
    model=model
)



triage_agent = Agent(
    name="TriageAgent",
    instructions="""
You are a smart triage agent for a todo app. Analyze the user's request and decide whether to create, read, update, or delete todos.

➡️ Handoff based on intent:
- 'create', 'add', 'new' → CreateAgent
- 'list', 'show', 'view' → ReadAgent
- 'update', 'edit', 'modify' → UpdateAgent
- 'delete', 'remove', 'clear' → DeleteAgent

📦 Extract and pass parameters as needed to the destination agent:
- If user says "delete all todos", pass `delete_all=True`.
- If user says "update all todos", pass `update_all=True`.
- If user says "generate creative title" or "make up a title", pass `auto_generate=True`.
- If they say "delete grocery todo", pass `title='grocery'`.
- If user gives partial title or description, use it to match.

🧠 Be smart and natural — the user might not use exact keywords. Infer intent and relevant fields for the agent.
""",
    handoffs=[create_agent, read_agent, update_agent, delete_agent],
    model=model
)


set_tracing_disabled(disabled=True)

# Main function to run the app
async def main():
    history = []
    while True:
        user_input = input("Enter your todo request (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        history.append({"role":"user", "content":user_input})
        # Pass history to the agent runner
        result = await Runner.run(triage_agent, history)
        print(f"Assistant: {result.final_output}")

        history=result.to_input_list()
if __name__ == "__main__":
    asyncio.run(main())
