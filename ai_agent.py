import chromadb #Chroma is saving memory data
from chromadb import PersistentClient
import requests
import time
from ollama import chat
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import streamlit as st
import os



def init_chromadb(storage_dir=".chroma_db"):
    # Connection to the chroma database engine
    client = PersistentClient(path=storage_dir)

    # Create a collection, now. collection is like a named folder
    # inside a database
    collection = client.get_or_create_collection(
        name="book_memory",
        metadata={"project": "book_finder_agent"}
    )

    # Finally, return client and collections, we can use them in
    # other parts of the project. save and recall data.
    return client, collection

def extract_topic_from_query(user_message):
    """
    Uses LLM to extract the core topic of the user's query.
    The LLM is NOT generating books, only extracting a subject/topic.
    """

    prompt = f"""
    Extract only the main topic or subject from this user query. 
    Return 1–4 words only. 
    No sentences. No extra text.

    Query: "{user_message}"
    Topic:
    """

    try:
        response = chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
        topic = response["message"]["content"].strip()
        return topic
    except Exception as e:
        print("Topic extraction failed:", e)
        return user_message   # fallback to raw query
    
    

def fetch_openlibrary_by_title_or_subject(query, limit=5):
    """
    search OpenLibrary for books matching the query.
    Returns a list of dictionaries with title, author, year, and description.
     
    """
    # API endpoint for search
    topic = extract_topic_from_query(query)
    
    url = "https://www.googleapis.com/books/v1/volumes"
    
    search_query = f"title:({topic}) OR subject:({topic})"

    # Query parameters
    params = {"q": search_query, "maxResults": limit}

    # Make a request to the OpenLibrary API
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        print("Google Books Error:", e)
        return []

    items = data.get("items", [])
    results = []

    for item in items:
        volume = item.get("volumeInfo", {})

        title = volume.get("title") or "Unknown Title"
        authors = volume.get("authors", [])
        author = ", ".join(authors) if authors else "Unknown"
        year = volume.get("publishedDate", "")
        desc = volume.get("description", "")[:300]  # trimmed
        link = volume.get("infoLink", "")

        results.append({
            "title": title,
            "author": author,
            "year": year,
            "desc": desc,
            "link": link  # adding link for richer formatting support
        })

    return results

#Formatting raw results from the model
def pretty_raw_results(results):
    """
    Convert the list of book results into readable text,
    One line per book
    """
    lines = []
    for r in results:
        title = r.get('title', 'Unknown')
        author = r.get('author', 'Unknown')
        desc = r.get('desc', '')
        year = r.get('year','unknown')
        link = r.get('link', 'unknown')
        line = f"{title} - {author} -{year} - {desc} - {link}"
        lines.append(line)
        
    return "\n".join(lines)    


#Teaching our agent to remember
def store_to_memory(client, collection, query, results, user_id=None):
    """
    Save search results into the chroma collection,
    tag each record with optional user_id for session filtering
    
    """
    docs, metadatas, ids = [], [], []
    for i, r in enumerate(results):
        doc_text = f"{r.get('title')} - {r.get('author')} - {r.get('desc')}"
        meta = {
            "query": query,
            "title": r.get("title"),
            "year": str(r.get("year") or ""),
            "source": "openlibrary"
            
        }
        if user_id:
            meta["user_id"] = user_id
            
        docs.append(doc_text)
        metadatas.append(meta)
        ids.append(f"{user_id or 'anon'} - {int(time.time())} -{i}")
        
        

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    try:
        if hasattr(client, "persist"):
            client.persist()
    except Exception:
        pass

    
def retrieve_similar(collection, text, n=3, user_id=None):
    """
    Query the collection for documents semantically similar to text,
    optional user_id filters results to a single user's memory.
    """
    try:
        if user_id:
            res = collection.query(query_texts=[text], n_results=n, where={"user_id": user_id})
        else:
            res = collection.query(query_texts=[text], n_results=n)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        combined = []

        for doc, meta in zip(docs, metas):
            combined.append({
                "title": meta.get("title"),
                "author": meta.get("author"),
                "doc": doc
            })

        return combined
    except Exception:
        return []


#Connect to ollama

def call_ollama_direct(prompt_text):
    """
    Send the prompt to Ollama and return the model's reply.
    """
    messages = [{"role": "user", "content": prompt_text}]
    try:
        response = chat(model="tinyllama", messages=messages)
    except Exception as e:
        return f"Error calling Ollama: {e}"

    if isinstance(response, dict):
        msg = response.get("message", {})
        if isinstance(msg, dict):
            return msg.get("content", "") or str(response)
        return str(response)
    else:
        try:
            return response.message.content
        except Exception:
            return str(response)



INTENT_TEMPLATE_STR = """
You are an assistant whose job is to identify the user's intent.
Given the message below, return a short phrase describing what they want.

User message:
{user_message}
"""

FORMAT_TEMPLATE_STR = """
You are a friendly book guide.
Using the raw book results below, create a clear and beginner friendly reply.

Rules:
- Return up to three books.
- Each line must contain: Title (as a Markdown link), the author, and a one-line summary.
- Preserve any links provided in the raw results: do not remove or rewrite them.
- If a result has no link, still include title and summary.
- Use ONLY the books listed in "Raw results" below. Do NOT invent or reuse books from the example.
- Return only Markdown; do not return JSON or plain text explanation.

Raw results (JSON-like):
{raw_results}

Example output (Markdown):
1. [Deep Learning](https://books.google.com/...)(year) — Ian Goodfellow — A comprehensive textbook about deep learning...
2. [Python Crash Course](https://books.google.com/...)(year) — Eric Matthes — A hands-on introduction...
"""


intent_prompt = PromptTemplate.from_template(INTENT_TEMPLATE_STR)
format_prompt = PromptTemplate.from_template(FORMAT_TEMPLATE_STR)

intent_chain = RunnableLambda(
    lambda user_message: intent_prompt.format(user_message=user_message)
) | RunnableLambda(
    lambda prompt_text: call_ollama_direct(prompt_text)
)

format_chain = RunnableLambda(
    lambda raw_text: format_prompt.format(raw_results=raw_text)
) | RunnableLambda(
    lambda prompt_text: call_ollama_direct(prompt_text)
)

def extract_intent(user_message):
    result = intent_chain.invoke(user_message)
    return str(result).strip()

def format_recommendations(raw_results):
    cleaned  =pretty_raw_results(raw_results)
    result = format_chain.invoke(cleaned)
    return str(result).strip()





def planner_decide_tools(intent_text):
    lowered = intent_text.lower()
    if 'book' in lowered or 'recommend' in lowered or 'similar' in lowered or 'novel' in lowered:
        return {'use_openlibrary': True}
    return {'use_openlibrary': True}



def run_agent_once(user_message, client=None, collection=None, user_id=None, max_hits=5):
    """
    Full agent pipeline for a single user message.
    Extract intent, plan the next step, call the tool,
    store the memory, and return the final formatted answer.
    """
    intent = extract_intent(user_message)
    plan = planner_decide_tools(intent)

    raw_results = []
    if plan.get("use_openlibrary"):
        raw_results = fetch_openlibrary_by_title_or_subject(user_message, limit=max_hits)
        print("RAW RESULTS:", raw_results)

    if client and collection and raw_results:
        store_to_memory(client, collection, user_message, raw_results, user_id=user_id)

    final_message = format_recommendations(raw_results)

    return {
        "intent": intent,
        "plan": plan,
        "raw_results": raw_results,
        "formatted": final_message
    }
    
def save_chat_message(client, collection, role, text, user_id=None):
    """
    Save a single chat message to the Chroma collection.
    Metadata: role ('user' or 'agent'), ts (timestamp), source='chat', optional user_id.
    """
    if not collection:
        return
    ts = float(time.time())
    # create an id unique enough: chat_userid_timestamp_ms_rand
    doc_id = f"chat_{user_id or 'anon'}_{int(ts*1000)}"
    try:
        collection.add(
            documents=[text],
            metadatas=[{
                "role": role,
                "ts": ts,
                "source": "chat",
                **({"user_id": user_id} if user_id else {})
            }],
            ids=[doc_id]
        )
        # persist if client supports it
        try:
            if hasattr(client, "persist"):
                client.persist()
        except Exception:
            pass
    except Exception as e:
        # non-fatal: log to console so UI doesn't break
        print("Error saving chat message to Chroma:", e)
        
        
def load_user_chat(collection, user_id):
    """
    Load all chat messages for given user_id from Chroma collection.
    Returns a list of (role, text) ordered by ts ascending.
    Looks for metadata 'source' == 'chat' or 'role' in metadata.
    """
    if not collection or not user_id:
        return []

    docs = []
    metas = []

    # Try preferred 'get' API first
    try:
        res = collection.get(where={"user_id": user_id})
        # some Chroma clients return dict with 'documents' and 'metadatas'
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
    except Exception:
        # fallback: large query filtered by metadata
        try:
            res = collection.query(query_texts=[""], n_results=1000, where={"user_id": user_id})
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
        except Exception as e:
            print("Error loading chat (fallback):", e)
            return []

    # sanitize shapes
    if not isinstance(docs, list) or not isinstance(metas, list):
        return []

    combined = []
    for doc, meta in zip(docs, metas):
        if not isinstance(meta, dict):
            continue
        # accept chat items marked with source='chat' OR those having a role field
        if meta.get("source") == "chat" or meta.get("role") in ("user", "agent"):
            try:
                ts = float(meta.get("ts", 0))
            except Exception:
                ts = 0.0
            role = meta.get("role", "agent" if meta.get("source") != "chat" else "user")
            # doc might be a string, but sometimes nested lists; coerce
            if isinstance(doc, list):
                # some query outputs nest results; flatten join
                doc_text = " ".join(doc)
            else:
                doc_text = str(doc)
            combined.append((ts, role, doc_text))

    # sort by timestamp ascending and return (role, text) list
    combined.sort(key=lambda x: x[0])
    history = [(role, text) for (_ts, role, text) in combined]
    return history


#Run when he user submit prompt
#Run when the user submits prompt
def submit():
    client = st.session_state.client
    collection = st.session_state.collection

    user_message = st.session_state["user_input"]

    if user_message:
        user_message = st.session_state['user_input']
    if user_message:
        
        if 'active_profile' not in st.session_state or st.session_state.active_profile != username:
            st.session_state.active_profile = username
        
        # load saved chat for this username and populate session history
        
        if username:
                CHROMA_DIR = f".chroma_db_{username}"
                client, collection = init_chromadb(CHROMA_DIR)
        else:
                CHROMA_DIR = ".chroma_db"
                client, collection = init_chromadb(CHROMA_DIR)
                
        st.session_state.history.append(("user", user_message))

        # Save user's message
        save_chat_message(
            client,
            collection,
            role="user",
            text=user_message,
            user_id=username
        )

        # Run the agent
        result = run_agent_once(
            user_message,
            client,
            collection,
            user_id=username
        )

        st.session_state.history.append(("agent", result["formatted"]))

        # Save agent's reply
        save_chat_message(
            client,
            collection,
            role="agent",
            text=result["formatted"],
            user_id=username
        )

        if "empty_key" not in st.session_state:
            st.session_state.empty_key = ""

        st.session_state.empty_key = st.session_state.user_input
        st.session_state.user_input = ""
        
st.set_page_config(page_title="AI Book Finder Agent", layout="wide")
st.title("AI Book Finder Agent")

st.sidebar.header("Session")
username = st.sidebar.text_input(
    "Enter your profile name",
    value="Default_user"
)
username = (username or "").strip() or "Default_user"

# Initialize history once
if "history" not in st.session_state:
    st.session_state.history = []

# Create/Open Chroma database every run
if username:
    CHROMA_DIR = f".chroma_db_{username}"
else:
    CHROMA_DIR = ".chroma_db"

client, collection = init_chromadb(CHROMA_DIR)

# Save them so submit() can access them
st.session_state.client = client
st.session_state.collection = collection

# Only reload chat when switching users
if (
    "active_profile" not in st.session_state
    or st.session_state.active_profile != username
):
    st.session_state.active_profile = username

    try:
        loaded_history = load_user_chat(collection, username)
        st.session_state.history = loaded_history if loaded_history else []
    except Exception as e:
        print("Error loading user chat:", e)
        st.session_state.history = []

chat_container = st.container()

with chat_container:
    for role, text in st.session_state.history:

        if role == "user":
            st.markdown(
                f"<div style='text-align:right; color:white; background-color:#2b7a78; "
                f"padding:8px; border-radius:6px; margin:4px 0;'>{text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:left; color:#222; background-color:#d9d9d9; "
                f"padding:8px; border-radius:6px; margin:4px 0;'>{text}</div>",
                unsafe_allow_html=True,
            )

user_message = st.text_input(
    "Type your message",
    key="user_input",
    on_change=submit,
)