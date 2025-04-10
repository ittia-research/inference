import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field

import torch
import torch.nn.functional as F
from torch import Tensor
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModel, AutoConfig

from core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name, max_length=settings.EMBEDDING_MAX_LENGTH, batch_size=settings.EMBEDDING_BATCH_SIZE):
        logger.info("Initializing EmbeddingModel...")
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.init_device()  # Should be after model_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        logger.info(f"Loading model: {model_name} to device: {self.device}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype='auto').to(self.device)

        logger.info(f"Model loaded successfully on device: {self.device}")

        self.bucket_ranges = self.generate_bucket_ranges()
        
        self._warmup()

    def init_device(self):
        """Initializes the device and sets up the environment for TPU.

        TO-DO: shall we set random seed?

        Reference: https://github.com/vllm-project/vllm/blob/790b79750b596043036b9fcbee885827fdd2ef3d/vllm/worker/tpu_worker.py
        """
        logger.info("Initializing device and environment...")

        torch.set_grad_enabled(False)

        # TO-DO: check if this works
        torch.set_default_dtype(self.model_config.torch_dtype)
        # torch.set_default_dtype(torch.bfloat16)
        print(f"Using the model dtype: {self.model_config.torch_dtype}")

        self.device = xm.xla_device()
        logger.info(f"Using device: {self.device}")
            
        try:
            rank = xr.global_ordinal()
            per_rank_path = os.path.join(settings.XLA_CACHE_DIR, f"rank_{rank}")
            os.makedirs(per_rank_path, exist_ok=True)
            xr.initialize_cache(per_rank_path, readonly=False)
            logger.info(f"XLA persistent cache initialized at: {per_rank_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize XLA persistent cache: {e}")


    def get_tokenized_inputs(self, input_texts, max_length):
        try:
            # With truncation = False, longer text will preserve but process will be much slower
            # padding='max_length'  # TO-DO
            batch_dict = self.tokenizer(input_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            tokenized = {k: v.to(self.device) for k, v in batch_dict.items()}
            return tokenized, batch_dict  # TPU, CPU
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise

    def embed_queries(self, queries):
        """Generates embeddings for queries."""
        queries = [self.get_detailed_instruct(q) for q in queries]
        return self.embed_docs(queries)

    def embed_docs(self, documents):
        """Generates embeddings for documents, now processes in batches.

        TO-DO: shall we dynamic batch size limit also?
        """
        ordered_embeddings = [None] * len(documents) # list to store embeddings in order
        
        # Pre-tokenize documents and assign to buckets
        document_batches = {} # Use this to collect document indices within each bucket
        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenizer.encode(doc, truncation=False)
            token_length = len(tokens)
            assigned_bucket = None
            for start, end in self.bucket_ranges:
                if start <= token_length <= end or (start, end) == self.bucket_ranges[-1]:  # Very long docs fallback to the last backet
                    assigned_bucket = (start, end)
                    break
            if assigned_bucket not in document_batches:
                document_batches[assigned_bucket] = []
            document_batches[assigned_bucket].append(doc_idx) # Store index instead of doc for efficiency


        for bucket_range, doc_indices in document_batches.items():
            max_len_in_batch = bucket_range[1]  # Static max_length to lower the need to compile
            bucket_docs = [documents[idx] for idx in doc_indices] # Fetch documents by index
            for i in range(0, len(bucket_docs), self.batch_size):
                batch_doc_indices = doc_indices[i:i + self.batch_size] # Indices for the current batch
                batch_docs = bucket_docs[i:i + self.batch_size] # Docs for the current batch
                try:
                    logger.debug(f"Processing bucket {bucket_range}, batch {i//self.batch_size + 1}/{len(bucket_docs)//self.batch_size + (1 if len(bucket_docs)%self.batch_size>0 else 0)}, batch size: {len(batch_docs)}, max_len: {max_len_in_batch}")

                    normalize_embeddings = self._direct_embed_batch(batch_docs, max_len_in_batch)
                    
                    for batch_index, original_doc_index in enumerate(batch_doc_indices):
                        ordered_embeddings[original_doc_index] = normalize_embeddings[batch_index] # Store embedding in correct order

                    logger.debug(f"Batch {i//self.batch_size + 1} embeddings generated for bucket {bucket_range}")


                except Exception as e:
                    logger.error(f"Error generating embeddings for bucket {bucket_range}, batch {i//self.batch_size + 1}: {e}")
                    raise # Re-raise to let FastAPI handle error response

        # return torch.cat(all_embeddings, dim=0) if all_embeddings else all_embeddings
        return torch.stack(ordered_embeddings)

    def _direct_embed_batch(self, batch_docs, max_len_in_batch):
        """Generate embeddings directly with batch limits"""
        tokenized, tokenized_cpu = self.get_tokenized_inputs(batch_docs, max_len_in_batch)
        logger.debug(f"Input batch shape: {tokenized_cpu['input_ids'].shape}") # Shape Debugging
        
        outputs = self.model(**tokenized)
        embeddings = self.last_token_pool(outputs.last_hidden_state, tokenized_cpu['attention_mask'])
        normalize_embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
        return normalize_embeddings

    def calculate_scores(self, query_embeddings, document_embeddings):
        """Calculates similarity scores."""
        try:
            scores = (query_embeddings @ document_embeddings.T) * 100
            return scores
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            raise

    def get_similarity_scores(self, queries, documents):
        """Generates embeddings and calculates similarity scores."""
        try:
            logger.info("Generating query embeddings...")
            query_embeddings = self.embed_queries(queries)
            logger.info("Generating document embeddings...")
            document_embeddings = self.embed_docs(documents)
            logger.info("Calculating scores...")
            scores = self.calculate_scores(query_embeddings, document_embeddings)
            return scores
        except Exception as e:
            logger.error(f"Error in get_similarity_scores: {e}")
            raise

    def get_detailed_instruct(self, query: str, task_description: str = None) -> str:
        if not task_description:
            task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        return f'Instruct: {task_description}\nQuery: {query}'

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
    def _warmup(self):
        """Performs a warmup run to trigger initial compilation for each bucket range and batch size."""
        logger.info("Performing warmup run...")
        dummy_text = ["Dummy document for warmup."]
        for start, end in self.bucket_ranges:
            for batch in range(1, self.batch_size + 1):
                try:
                    _ = self._direct_embed_batch(dummy_text * batch, end)
                    logger.info(f"Warmup: token length range {start} ~ {end} batch {batch}")
                except Exception as e:
                    logger.warning(f"Warmup process failed: {e}")
        logger.info("Warmup run completed.")

    def generate_bucket_ranges(self):
        """Generates bucket ranges up to max_length."""
        step = 256
        start = 0
        ranges = []
        while start < self.max_length:
            end = min(start + step, self.max_length)
            ranges.append((start, end))
            start = end + 1
        return ranges


app = FastAPI(title="Embedding API", description="API for generating text embeddings on TPU.")

# --- Dependency to get the EmbeddingModel ---
async def get_embedding_model() -> EmbeddingModel:
    return app.state.embedding_model

# --- Data Models for Request and Response ---
class SimilarityRequest(BaseModel):
    queries: List[str] = Field(..., description="List of query strings")
    documents: List[str] = Field(..., description="List of document strings")

class SimilarityResponse(BaseModel):
    scores: List[List[float]] = Field(..., description="Similarity scores matrix (queries x documents)")

# OpenAI API Compatible Models
class OpenAIEmbeddingsRequest(BaseModel):
    model: Optional[str] = Field("gte-qwen2-1.5b-instruct", description="Model name (optional, can be ignored)")
    input: List[str] = Field(..., description="List of input strings to embed")
    encoding_format: Optional[str] = Field("float", description="Encoding format (optional, defaults to float)")

class EmbeddingData(BaseModel):
    object: str = Field("embedding", description="Object type")
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="Index of the embedding in the list")

class Usage(BaseModel):
    prompt_tokens: int = Field(0, description="Number of prompt tokens used (mocked value)") # Mocked for now
    total_tokens: int = Field(0, description="Number of total tokens used (mocked value)") # Mocked for now

class OpenAIEmbeddingsResponse(BaseModel):
    object: str = Field("list", description="Object type, always 'list' for embeddings")
    data: List[EmbeddingData] = Field(..., description="List of embedding objects")
    model: str = Field("gte-qwen2-1.5b-instruct", description="Model name (echoed from request or default)") # Echoed or default
    usage: Usage = Field(..., description="Token usage information")


# --- Startup Event to Load the Model ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI application and loading EmbeddingModel...")
    try:
        app.state.embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL_ID) # Store in app state
        logger.info("EmbeddingModel loaded successfully into application state.")
    except Exception as e:
        logger.error(f"Error loading EmbeddingModel during startup: {e}")
        raise  # Prevent the FastAPI app from starting if model fails to load


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint to verify API availability."""
    return {"status": "ok"}


# --- Similarity Endpoint ---
@app.post("/similarity", response_model=SimilarityResponse, status_code=status.HTTP_200_OK)
async def get_similarity(request: SimilarityRequest, model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Calculates similarity scores between queries and documents.
    """
    try:
        scores_tensor = model.get_similarity_scores(request.queries, request.documents)
        scores_list = scores_tensor.tolist() # Convert tensor to list for JSON serialization
        return {"scores": scores_list}
    except Exception as e:
        logger.error(f"Error processing similarity request: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error calculating similarity scores")


# --- Embeddings Endpoint (OpenAI API Compatible) ---
@app.post("/embeddings", response_model=OpenAIEmbeddingsResponse, status_code=status.HTTP_200_OK)
async def get_openai_embeddings(request: OpenAIEmbeddingsRequest, model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Generates embeddings for a list of texts in OpenAI API compatible format.
    """
    try:
        embeddings_tensor = model.embed_docs(request.input)
        embeddings_list = embeddings_tensor.tolist()

        # Format the response to match OpenAI API
        embedding_data_list = []
        for index, embedding_vector in enumerate(embeddings_list):
            embedding_data_list.append(EmbeddingData(embedding=embedding_vector, index=index))

        # Mock usage data
        usage_data = Usage(prompt_tokens=0, total_tokens=0)

        return OpenAIEmbeddingsResponse(
            data=embedding_data_list,
            model=request.model if request.model else "gte-qwen2-1.5b-instruct",
            object="list",
            usage=usage_data
        )
    except Exception as e:
        logger.error(f"Error processing embeddings request: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating embeddings")
