import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
from typing import List, Dict, Any
from torch.utils.data import DataLoader, Dataset
from functools import partial

# Configure GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# Define subset size
SUBSET_SIZE = 500  # Starting with 500 items for quick testing

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

def generate_case_study(row: Dict[str, Any]) -> str:
    """Generate a detailed case study for a repository using available metadata"""
    # Extract relevant information from the row
    summary = row.get('summary', '').strip()
    docstring = row.get('docstring', '').strip()
    repo_name = row.get('repo', '').strip()
    
    # Generate a more detailed overview using available information
    overview = summary if summary else "This repository provides a software implementation"
    if docstring:
        # Extract the first paragraph of the docstring for additional context
        first_para = docstring.split('\n\n')[0].strip()
        overview = f"{overview}. {first_para}"
    
    # Analyze the repository path to infer technology stack
    path_components = row.get('path', '').lower().split('/')
    tech_stack = []
    
    # Common technology indicators in paths
    if any('python' in comp for comp in path_components):
        tech_stack.append("Python")
    if any('tensorflow' in comp or 'tf' in comp for comp in path_components):
        tech_stack.append("TensorFlow")
    if any('pytorch' in comp for comp in path_components):
        tech_stack.append("PyTorch")
    if any('react' in comp for comp in path_components):
        tech_stack.append("React")
    
    tech_stack_str = ", ".join(tech_stack) if tech_stack else "various technologies"
    
    case_study = f"""
### Overview
{overview}

### Technical Implementation
This project is built using {tech_stack_str}. The implementation focuses on providing a robust and maintainable solution for {summary.lower() if summary else 'the specified requirements'}.

### Key Features
- Primary functionality: {summary if summary else 'Implementation of core project requirements'}
- Complete documentation and code examples
- Well-structured implementation following best practices
- Modular design for easy integration and customization

### Use Cases
This repository is particularly valuable for:
- Developers implementing similar functionality in their projects
- Teams looking for reference implementations and best practices
- Projects requiring similar technical capabilities
- Learning and educational purposes in related technical domains

### Integration Considerations
The repository can be integrated into existing projects, with consideration for:
- Compatibility with existing technology stacks
- Required dependencies and prerequisites
- Potential customization needs
- Performance and scalability requirements
    """
    return case_study

def display_recommendations(recommendations: pd.DataFrame):
    """Display recommendations in a list format with all details"""
    st.markdown("### 🎯 Top Recommendations")
    
    # Create a list of recommendations
    for idx, row in recommendations.iterrows():
        with st.container():
            # Header with repository name and match score
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {idx + 1}. {row['repo']}")
            with col2:
                st.metric("Match Score", f"{row['similarity']:.2%}")
            
            # Repository details
            st.markdown(f"**URL:** [View Repository]({row['url']})")
            st.markdown(f"**Path:** `{row['path']}`")
            
            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("👍", key=f"like_{idx}"):
                    st.session_state.feedback[row['repo']] = st.session_state.feedback.get(row['repo'], {'likes': 0, 'dislikes': 0})
                    st.session_state.feedback[row['repo']]['likes'] += 1
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎", key=f"dislike_{idx}"):
                    st.session_state.feedback[row['repo']] = st.session_state.feedback.get(row['repo'], {'likes': 0, 'dislikes': 0})
                    st.session_state.feedback[row['repo']]['dislikes'] += 1
                    st.success("Thanks for your feedback!")
            
            # Documentation and case study in tabs
            tab1, tab2 = st.tabs(["📚 Documentation", "📑 Case Study"])
            with tab1:
                if row['docstring']:
                    st.markdown(row['docstring'])
                else:
                    st.info("No documentation available")
            
            with tab2:
                st.markdown(generate_case_study(row))
            
            st.markdown("---")

@st.cache_resource
def load_data_and_model():
    """Load the dataset and model with optimized memory usage"""
    try:
        # Load dataset
        dataset = load_dataset("frankjosh/filtered_dataset")
        data = pd.DataFrame(dataset['train'])
        
        # Take a random subset
        data = data.sample(n=min(SUBSET_SIZE, len(data)), random_state=42).reset_index(drop=True)
        
        # Combine text fields
        data['text'] = data['docstring'].fillna('') + ' ' + data['summary'].fillna('')
        
        # Load model and tokenizer
        model_name = "Salesforce/codet5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to(device)
        
        model.eval()
        return data, tokenizer, model
    
    except Exception as e:
        st.error(f"Error in initialization: {str(e)}")
        st.stop()

def collate_fn(batch, pad_token_id):
    max_length = max(inputs['input_ids'].shape[1] for inputs in batch)
    input_ids = []
    attention_mask = []
    
    for inputs in batch:
        input_ids.append(torch.nn.functional.pad(
            inputs['input_ids'].squeeze(),
            (0, max_length - inputs['input_ids'].shape[1]),
            value=pad_token_id
        ))
        attention_mask.append(torch.nn.functional.pad(
            inputs['attention_mask'].squeeze(),
            (0, max_length - inputs['attention_mask'].shape[1]),
            value=0
        ))
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask)
    }

def generate_embeddings_batch(model, batch, device):
    """Generate embeddings for a batch of inputs"""
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.encoder(**batch)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

def precompute_embeddings(data: pd.DataFrame, model, tokenizer, batch_size: int = 16):
    """Precompute embeddings with batching and progress tracking"""
    dataset = TextDataset(data['text'].tolist(), tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True
    )
    
    embeddings = []
    total_batches = len(dataloader)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = datetime.now()
    
    for i, batch in enumerate(dataloader):
        # Generate embeddings for batch
        batch_embeddings = generate_embeddings_batch(model, batch, device)
        embeddings.extend(batch_embeddings)
        
        # Update progress
        progress = (i + 1) / total_batches
        progress_bar.progress(progress)
        
        # Calculate and display ETA
        elapsed_time = (datetime.now() - start_time).total_seconds()
        eta = (elapsed_time / (i + 1)) * (total_batches - (i + 1))
        status_text.text(f"Processing batch {i+1}/{total_batches}. ETA: {int(eta)} seconds")
    
    progress_bar.empty()
    status_text.empty()
    
    # Add embeddings to dataframe
    data['embedding'] = embeddings
    return data

@torch.no_grad()
def generate_query_embedding(model, tokenizer, query: str) -> np.ndarray:
    """Generate embedding for a single query"""
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    outputs = model.encoder(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding.squeeze()

def find_similar_repos(query_embedding: np.ndarray, data: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Find similar repositories using vectorized operations"""
    similarities = cosine_similarity([query_embedding], np.stack(data['embedding'].values))[0]
    data['similarity'] = similarities
    return data.nlargest(top_n, 'similarity')

# Load resources
data, tokenizer, model = load_data_and_model()

# Add info about subset size
st.info(f"Running with a subset of {SUBSET_SIZE} repositories for testing purposes.")

# Precompute embeddings for the subset
data = precompute_embeddings(data, model, tokenizer)

# Main App Interface
st.title("Repository Recommender System 🚀")
st.caption("Testing Version - Running on subset of data")

# Main interface
user_query = st.text_area(
    "Describe your project:",
    height=150,
    placeholder="Example: I need a machine learning project for customer churn prediction..."
)

# Search button and filters
col1, col2 = st.columns([2, 1])
with col1:
    search_button = st.button("🔍 Search Repositories", type="primary")
with col2:
    top_n = st.selectbox("Number of results:", [3, 5, 10], index=1)

if search_button and user_query.strip():
    with st.spinner("Finding relevant repositories..."):
        # Generate query embedding and get recommendations
        query_embedding = generate_query_embedding(model, tokenizer, user_query)
        recommendations = find_similar_repos(query_embedding, data, top_n)
        
        # Save to history
        st.session_state.history.append({
            'query': user_query,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': recommendations['repo'].tolist()
        })
        
        # Display recommendations using the new function
        display_recommendations(recommendations)

# Sidebar for History and Stats
with st.sidebar:
    st.header("📊 Search History")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"**Search {len(st.session_state.history)-idx}**")
            st.markdown(f"Query: _{item['query'][:30]}..._")
            st.caption(f"Time: {item['timestamp']}")
            st.caption(f"Results: {len(item['results'])} repositories")
            if st.button("Rerun this search", key=f"rerun_{idx}"):
                st.session_state.rerun_query = item['query']
            st.markdown("---")
    else:
        st.write("No search history yet")

    st.header("📈 Usage Statistics")
    st.write(f"Total Searches: {len(st.session_state.history)}")
    if st.session_state.feedback:
        feedback_df = pd.DataFrame(st.session_state.feedback).T
        feedback_df['Total'] = feedback_df['likes'] + feedback_df['dislikes']
        st.bar_chart(feedback_df[['likes', 'dislikes']])

# Footer
st.markdown("---")
st.markdown(
    """
    Made with 🤖 using CodeT5 and Streamlit |
    GPU Status: {'🟢 Enabled' if torch.cuda.is_available() else '🔴 Disabled'} |
    Model: CodeT5-Small
    """
)