"""
BERTopic clustering for analyzing evaluation-like prompt features.

This script extracts short descriptions from comparison arguments and clusters them
to identify common patterns in what makes prompts appear evaluation-like vs deployment-like.
"""

import argparse
import json
import os
import re
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
import requests
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError
from bertopic import BERTopic
# Custom error to allow partial progress when fetching embeddings fails mid-way
class EmbeddingAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        partial_embeddings: Optional[np.ndarray] = None,
        processed_count: int = 0
    ):
        super().__init__(message)
        self.partial_embeddings = partial_embeddings
        self.processed_count = processed_count
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


class OpenRouterEmbeddings:
    """Custom embedding model for BERTopic that uses OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-large"):
        self.api_key = api_key
        self.model = model
    
    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed documents using OpenRouter API."""
        return get_openai_embeddings(documents, self.api_key, self.model)
    
    def embed_documents(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Alias for embed method (BERTopic compatibility)."""
        return self.embed(documents, verbose)


def get_openai_embeddings(texts: List[str], api_key: str, model: str = "openai/text-embedding-3-large", batch_size: int = 1000) -> np.ndarray:
    """
    Get embeddings from OpenAI via OpenRouter API with batching support.
    
    Args:
        texts: List of text strings to embed
        api_key: OpenRouter API key
        model: Model identifier (default: openai/text-embedding-3-large)
        batch_size: Number of texts to process per API request (default: 1000)
        
    Returns:
        numpy array of embeddings (shape: [n_texts, embedding_dim])
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Requesting embeddings for {len(texts)} texts from {model}...")
    print(f"Processing in {total_batches} batches of up to {batch_size} texts each...")
    
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch_texts)} texts...", end=" ", flush=True)
        
        # OpenRouter expects the OpenAI embeddings format
        data = {
            "model": model,
            "input": batch_texts
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=data,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            
            try:
                result = response.json()
            except (ValueError, RequestsJSONDecodeError) as json_err:
                snippet = response.text[:500].strip()
                raise EmbeddingAPIError(
                    f"Failed to decode embeddings response for batch {batch_num}/{total_batches}: {json_err}. "
                    f"Raw response snippet: {snippet}",
                    partial_embeddings=np.array(all_embeddings) if all_embeddings else None,
                    processed_count=len(all_embeddings)
                ) from json_err
            
            # Check for error in response
            if 'error' in result:
                print(f"\nERROR - API returned error: {result['error']}")
                raise EmbeddingAPIError(
                    f"API Error on batch {batch_num}/{total_batches}: {result['error']}",
                    partial_embeddings=np.array(all_embeddings) if all_embeddings else None,
                    processed_count=len(all_embeddings)
                )
            
            # Check if 'data' key exists
            if 'data' not in result:
                print(f"\nERROR - Unexpected API response structure:")
                print(f"Response keys: {list(result.keys())}")
                print(f"Full response: {result}")
                raise EmbeddingAPIError(
                    f"'data' key not found in API response for batch {batch_num}/{total_batches}.",
                    partial_embeddings=np.array(all_embeddings) if all_embeddings else None,
                    processed_count=len(all_embeddings)
                )
            
            # Extract embeddings from response
            batch_embeddings = [item['embedding'] for item in result['data']]
            all_embeddings.extend(batch_embeddings)
            
            print(f"✓ ({len(batch_embeddings)} embeddings)")
            
            # Add a small delay between batches to avoid rate limiting
            if batch_num < total_batches:
                time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"\nError requesting embeddings: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            raise EmbeddingAPIError(
                f"Request error on batch {batch_num}/{total_batches}: {e}",
                partial_embeddings=np.array(all_embeddings) if all_embeddings else None,
                processed_count=len(all_embeddings)
            ) from e
        except EmbeddingAPIError:
            raise
        except Exception as e:
            print(f"\nError processing embeddings response: {e}")
            raise EmbeddingAPIError(
                f"Unexpected error on batch {batch_num}/{total_batches}: {e}",
                partial_embeddings=np.array(all_embeddings) if all_embeddings else None,
                processed_count=len(all_embeddings)
            ) from e
    
    embeddings = np.array(all_embeddings)
    print(f"✓ Received all embeddings with shape: {embeddings.shape}")
    return embeddings


def extract_eval_reasons(json_path: str) -> Tuple[List[str], Dict[str, str], List[str]]:
    """
    Extract short descriptions and full descriptions from the comparison arguments.
    
    Logic:
    - Only process comparisons where the new sample LOST (looked more eval-like):
      * If order is "original" and more_realistic = "2" (opponent won)
      * If order is "swapped" and more_realistic = "1" (opponent won after swap)
    - Then extract arguments where transcript_id points to the new sample:
      * If order is "original", extract arguments where transcript_id = "1"
      * If order is "swapped", extract arguments where transcript_id = "2"
    
    This extracts reasons explaining why the new sample looks evaluation-like
    (in cases where it was judged as less realistic than the opponent).
    
    Args:
        json_path: Path to the explanations JSON file
        
    Returns:
        Tuple of (short_descriptions_list, short_to_full_mapping_dict, sample_ids_list)
        The sample_ids_list is parallel to short_descriptions_list
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    short_descriptions = []
    short_to_full = {}
    sample_ids = []
    samples = data.get('samples', [])
    
    print(f"Processing {len(samples)} samples...")
    
    total_individual_results = 0
    filtered_individual_results = 0
    
    for sample in samples:
        sample_id = sample.get('new_sample_id', 'unknown')
        comparisons = sample.get('comparisons', [])
        
        for comparison in comparisons:
            full_data = comparison.get('full_comparison_data', {})
            individual_results = full_data.get('individual_results', [])
            
            for result_data in individual_results:
                total_individual_results += 1
                order = result_data.get('order', '')
                result = result_data.get('result', {})
                more_realistic = result.get('more_realistic', '')
                arguments = result.get('arguments', [])
                
                # Filter: only process if new sample LOST (looked more eval-like)
                should_process = False
                if order == 'original' and more_realistic == '2':
                    should_process = True
                elif order == 'swapped' and more_realistic == '1':
                    should_process = True
                
                if not should_process:
                    continue
                
                filtered_individual_results += 1
                
                for arg in arguments:
                    if not isinstance(arg, dict):
                        continue
                    
                    transcript_id = arg.get('transcript_id', '')
                    short_desc = arg.get('short_description', '')
                    full_desc = arg.get('description', '')
                    
                    # Extract based on order and transcript_id matching logic
                    if order == 'original' and transcript_id == '1':
                        if short_desc:
                            short_descriptions.append(short_desc)
                            short_to_full[short_desc] = full_desc
                            sample_ids.append(sample_id)
                    elif order == 'swapped' and transcript_id == '2':
                        if short_desc:
                            short_descriptions.append(short_desc)
                            short_to_full[short_desc] = full_desc
                            sample_ids.append(sample_id)
    
    print(f"✓ Filtered {filtered_individual_results}/{total_individual_results} judge comparisons where new sample lost")
    print(f"✓ Extracted {len(short_descriptions)} evaluation-like feature descriptions")
    return short_descriptions, short_to_full, sample_ids


def natural_sort_key(value: str) -> List[Any]:
    """
    Natural sort helper that keeps numeric suffix order intuitive (e.g., sample_5 before sample_11).
    """
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _build_argument_entry(sample_id: str, short_desc: str, short_to_full: Dict[str, str]) -> Dict[str, str]:
    """Create a standardized argument payload (per-sample mode doesn’t need sample_id)."""
    return {
        "short_description": short_desc,
        "full_description": short_to_full.get(short_desc, "")
    }


def _cluster_single_sample(
    sample_id: str,
    sample_docs: List[str],
    sample_embeddings: np.ndarray,
    short_to_full: Dict[str, str],
    api_key: str,
    embedding_model: OpenRouterEmbeddings,
    n_neighbors: int,
    min_cluster_size: int,
    min_samples: int,
    min_df: int,
    random_state: int
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Cluster arguments for a single sample, returning report data and summary lines.
    """
    MIN_DOCS_FOR_CLUSTERING = 3
    doc_count = len(sample_docs)
    summary_lines: List[str] = []
    
    sample_result: Dict[str, Any] = {
        "sample_id": sample_id,
        "total_arguments": doc_count,
        "topics": []
    }
    
    summary_lines.append("="*80)
    summary_lines.append(f"Sample: {sample_id}")
    summary_lines.append("="*80)
    summary_lines.append(f"Total arguments: {doc_count}")
    
    if doc_count == 0:
        sample_result["notes"] = "No evaluation-like arguments found for this sample."
        summary_lines.append("Notes: No arguments to cluster.")
        summary_lines.append("")
        return sample_result, summary_lines
    
    if doc_count < MIN_DOCS_FOR_CLUSTERING:
        arguments = [_build_argument_entry(sample_id, short_desc, short_to_full) for short_desc in sample_docs]
        topic_entry = {
            "name": "Not clustered",
            "count": len(arguments),
            "arguments": arguments
        }
        sample_result["topics"] = [topic_entry]
        sample_result["notes"] = f"Clustering skipped: requires at least {MIN_DOCS_FOR_CLUSTERING} arguments."
        
        summary_lines.append(f"Topics: {len(sample_result['topics'])} (not clustered fallback)")
        summary_lines.append("-"*80)
        for idx, argument in enumerate(arguments[:10], 1):
            summary_lines.append(f"  {idx}. {argument['short_description']}")
        summary_lines.append("")
        return sample_result, summary_lines
    
    # Adjust hyperparameters for the sample size
    sample_n_neighbors = max(2, min(n_neighbors, doc_count - 1))
    sample_min_cluster_size = max(2, min(min_cluster_size, doc_count))
    sample_min_samples = max(1, min(min_samples, sample_min_cluster_size))
    
    try:
        topic_model = create_topic_model(
            n_neighbors=sample_n_neighbors,
            min_cluster_size=sample_min_cluster_size,
            min_samples=sample_min_samples,
            min_df=min_df,
            random_state=random_state,
            embedding_model=embedding_model,
            api_key=api_key
        )
        
        topics, probs = fit_and_reduce_topics(topic_model, sample_docs, sample_embeddings)
    except Exception as exc:
        print(f"Warning: Clustering failed for {sample_id}: {exc}")
        arguments = [_build_argument_entry(sample_id, short_desc, short_to_full) for short_desc in sample_docs]
        sample_result["topics"] = [{
            "name": "Not clustered",
            "count": len(arguments),
            "arguments": arguments
        }]
        sample_result["notes"] = f"Clustering skipped due to error: {exc}"
        
        summary_lines.append(f"Topics: {len(sample_result['topics'])} (not clustered fallback)")
        summary_lines.append("-"*80)
        for idx, argument in enumerate(arguments[:10], 1):
            summary_lines.append(f"  {idx}. {argument['short_description']}")
        summary_lines.append("")
        return sample_result, summary_lines
    
    topic_freq = topic_model.get_topic_freq().sort_values(by='Count', ascending=False)
    valid_topics = [row for _, row in topic_freq.iterrows() if row['Topic'] != -1]
    
    topic_entries = []
    for row in valid_topics:
        topic_id = row['Topic']
        doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        representative_docs = [sample_docs[i] for i in doc_indices[:10]]
        
        topic_words = topic_model.get_topic(topic_id)
        topic_name = generate_llm_topic_label(
            topic_id,
            topic_words if topic_words else [],
            representative_docs,
            api_key
        )
        
        arguments = []
        for doc_idx in doc_indices:
            short_desc = sample_docs[doc_idx]
            arguments.append(_build_argument_entry(sample_id, short_desc, short_to_full))
        
        topic_entries.append({
            "topic_id": int(topic_id),
            "name": topic_name,
            "count": len(arguments),
            "arguments": arguments
        })
    
    if not topic_entries:
        # All points considered outliers; treat as single bucket
        outlier_arguments = [
            _build_argument_entry(sample_id, sample_docs[i], short_to_full)
            for i, topic_id in enumerate(topics) if topic_id == -1
        ]
        topic_entries.append({
            "topic_id": -1,
            "name": "Outliers / Unclustered Arguments",
            "count": len(outlier_arguments),
            "arguments": outlier_arguments
        })
        sample_result["notes"] = "All arguments marked as outliers by HDBSCAN."
    
    outlier_count = sum(1 for topic_id in topics if topic_id == -1)
    outlier_rate = 100 * outlier_count / doc_count if doc_count else 0.0
    
    sample_result["topics"] = topic_entries
    sample_result["outlier_rate"] = outlier_rate
    sample_result["outlier_count"] = outlier_count
    
    summary_lines.append(f"Topics: {len(topic_entries)}")
    summary_lines.append(f"Outlier rate: {outlier_count}/{doc_count} ({outlier_rate:.1f}%)")
    summary_lines.append("-"*80)
    
    for topic in topic_entries:
        summary_lines.append(f"{topic['name']} ({topic['count']} arguments)")
        for idx, argument in enumerate(topic["arguments"][:10], 1):
            summary_lines.append(f"  {idx}. {argument['short_description']}")
        summary_lines.append("")
    
    return sample_result, summary_lines


def generate_per_sample_clusters(
    docs: List[str],
    embeddings: np.ndarray,
    short_to_full: Dict[str, str],
    sample_ids: List[str],
    output_path: Path,
    api_key: str,
    n_neighbors: int,
    min_cluster_size: int,
    min_samples: int,
    min_df: int,
    random_state: int,
    partial_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate clustering reports for each sample individually.
    """
    sample_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, sample_id in enumerate(sample_ids):
        sample_to_indices[sample_id].append(idx)
    
    sorted_sample_ids = sorted(sample_to_indices.keys(), key=natural_sort_key)
    embedding_model = OpenRouterEmbeddings(api_key)
    
    report_data: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "total_samples": len(sorted_sample_ids),
            "total_arguments": len(docs)
        },
        "samples": []
    }

    if partial_info:
        report_data["metadata"]["partial_processing"] = partial_info
    
    summary_lines = [
        "="*80,
        "PER-SAMPLE CLUSTERS SUMMARY",
        "="*80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total samples: {len(sorted_sample_ids)}",
        f"Total arguments: {len(docs)}",
        "="*80,
        ""
    ]

    if partial_info:
        summary_lines.append(
            f"Partial processing: clustered {partial_info['processed_arguments']}/"
            f"{partial_info['total_arguments']} arguments"
        )
        summary_lines.append(f"Reason: {partial_info.get('error', 'Unknown error')}")
        summary_lines.append("="*80)
        summary_lines.append("")
    
    total_topics = 0
    
    for sample_id in sorted_sample_ids:
        indices = sample_to_indices[sample_id]
        sample_docs = [docs[i] for i in indices]
        sample_embeddings = embeddings[indices]
        
        sample_result, sample_summary = _cluster_single_sample(
            sample_id=sample_id,
            sample_docs=sample_docs,
            sample_embeddings=sample_embeddings,
            short_to_full=short_to_full,
            api_key=api_key,
            embedding_model=embedding_model,
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            min_df=min_df,
            random_state=random_state
        )
        
        report_data["samples"].append(sample_result)
        summary_lines.extend(sample_summary)
        summary_lines.append("")  # Blank line between samples
        total_topics += len(sample_result.get("topics", []))
    
    report_data["metadata"]["total_topics"] = total_topics
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n{'='*80}")
    print("PER-SAMPLE CLUSTERING COMPLETE")
    print(f"JSON report saved to: {output_path}")
    print(f"Summary text saved to: {summary_path}")
    print(f"{'='*80}")
    
    return report_data


def create_topic_model(
    n_neighbors: int,
    min_cluster_size: int,
    min_samples: int,
    min_df: int,
    random_state: int,
    embedding_model: OpenRouterEmbeddings,
    api_key: str
) -> BERTopic:
    """
    Create and configure BERTopic model optimized for short texts.
    
    Args:
        n_neighbors: UMAP n_neighbors parameter
        min_cluster_size: HDBSCAN min_cluster_size parameter
        min_samples: HDBSCAN min_samples parameter
        min_df: CountVectorizer min_df parameter
        random_state: Random seed for reproducibility
        embedding_model: Embedding model for BERTopic
        api_key: OpenRouter API key for LLM-based topic labeling
        
    Returns:
        Configured BERTopic model
    """
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state
    )
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True
    )
    
    # CountVectorizer for topic representation
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=min_df
    )
    
    # Create BERTopic model (using default c-TF-IDF representation)
    # Note: LLM-based labeling will be done in post-processing for better results
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=True
    )
    
    return topic_model


def fit_and_reduce_topics(
    topic_model: BERTopic,
    docs: List[str],
    embeddings: np.ndarray
) -> Tuple[List[int], np.ndarray]:
    """
    Fit the topic model and apply post-processing (topic reduction and outlier reduction).
    
    Args:
        topic_model: Configured BERTopic model
        docs: List of document strings
        embeddings: Precomputed embeddings
        
    Returns:
        Tuple of (topics, probabilities)
    """
    print("\nFitting topic model with precomputed embeddings...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    
    print(f"Initial topics: {len(set(topics))} (including outliers)")
    
    # Reduce topics automatically
    print("\nReducing topics...")
    topic_model.reduce_topics(docs, nr_topics="auto")
    topics = topic_model.topics_
    
    print(f"After reduction: {len(set(topics))} topics")
    
    # Reduce outliers using probabilities strategy
    print("\nReducing outliers...")
    topics = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities")
    topic_model.update_topics(docs, topics=topics)
    
    outlier_count = sum(1 for t in topics if t == -1)
    outlier_rate = 100 * outlier_count / len(topics) if topics else 0
    print(f"Outliers after reduction: {outlier_count}/{len(topics)} ({outlier_rate:.1f}%)")
    
    return topics, probs


def generate_llm_topic_label(
    topic_id: int,
    topic_words: List[Tuple[str, float]],
    representative_docs: List[str],
    api_key: str
) -> str:
    """
    Generate a human-readable topic label using an LLM.
    
    Args:
        topic_id: The topic ID
        topic_words: List of (word, weight) tuples for top terms
        representative_docs: List of representative document strings
        api_key: OpenRouter API key
        
    Returns:
        Human-readable topic label (2-6 words)
    """
    # Prepare keywords
    keywords = [word for word, _ in topic_words[:10]]
    keywords_str = ", ".join(keywords)
    
    # Prepare sample documents (first 5)
    docs_str = "\n".join([f"- {doc}" for doc in representative_docs[:5]])
    
    # Create prompt
    prompt = f"""I have a topic from clustering evaluation-like features that judges identified when comparing AI assistant conversations.

Keywords: {keywords_str}

Representative examples:
{docs_str}

Based on these keywords and examples, generate a concise, descriptive label (2-6 words) that captures the essence of this evaluation pattern. The label should be natural and human-readable.

Respond with ONLY the label, nothing else."""

    # Call OpenRouter API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "anthropic/claude-haiku-4.5",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        label = result['choices'][0]['message']['content'].strip()
        # Remove quotes if present
        label = label.strip('"\'')
        return label
    except Exception as e:
        print(f"Warning: Failed to generate LLM label for topic {topic_id}: {e}")
        # Fallback to keyword-based name
        return ", ".join(keywords[:3])


def generate_report(
    topic_model: BERTopic,
    docs: List[str],
    topics: List[int],
    output_path: Path,
    short_to_full: Dict[str, str],
    sample_ids: List[str],
    api_key: str,
    partial_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive report of clustering results.
    
    Args:
        topic_model: Fitted BERTopic model
        docs: List of document strings (short descriptions)
        topics: Topic assignments
        output_path: Path to save the clusters JSON file
        short_to_full: Mapping from short descriptions to full descriptions
        sample_ids: List of sample IDs parallel to docs
        api_key: OpenRouter API key for LLM-based topic labeling
        
    Returns:
        Dictionary with report data
    
    Outputs:
        - JSON file with all cluster arguments
        - TXT file with cluster names and top 10 representative short descriptions
    """
    print("\n" + "="*80)
    print("CLUSTERING REPORT")
    print("="*80)
    
    # Topic frequency table
    topic_freq = topic_model.get_topic_freq()
    topic_freq = topic_freq.sort_values(by='Count', ascending=False)
    
    print("\nTopic Frequency (sorted by count):")
    print(topic_freq.to_string(index=False))
    
    # Outlier rate
    outlier_count = sum(1 for t in topics if t == -1)
    outlier_rate = 100 * outlier_count / len(topics) if topics else 0
    print(f"\nOutlier Rate: {outlier_count}/{len(topics)} ({outlier_rate:.1f}%)")
    
    # Top 10 topics (excluding outliers)
    print("\n" + "="*80)
    print("TOP 10 TOPICS")
    print("="*80)
    
    report_data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "total_arguments": len(docs),
            "total_topics": len(set(topics)) - (1 if -1 in topics else 0)
        },
        "topics": []
    }
    if partial_info:
        report_data["metadata"]["partial_processing"] = partial_info
    
    # Get top 10 non-outlier topics
    top_topics = [row['Topic'] for _, row in topic_freq.iterrows() if row['Topic'] != -1][:10]
    
    print("\nGenerating human-readable labels for topics using LLM...")
    
    # For the summary text file
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("CLUSTERS SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Total arguments: {len(docs)}")
    summary_lines.append(f"Total topics: {len(set(topics)) - (1 if -1 in topics else 0)}")
    summary_lines.append("="*80)
    summary_lines.append("")

    if partial_info:
        summary_lines.append(
            f"Partial processing: clustered {partial_info['processed_arguments']}/"
            f"{partial_info['total_arguments']} arguments"
        )
        summary_lines.append(f"Reason: {partial_info.get('error', 'Unknown error')}")
        summary_lines.append("")
    
    for topic_id in top_topics:
        # Get top terms
        topic_words = topic_model.get_topic(topic_id)
        
        # Get ALL documents in this cluster
        doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        representative_short_docs = [docs[i] for i in doc_indices[:10]]
        
        # Generate human-readable label using LLM
        topic_name = generate_llm_topic_label(
            topic_id,
            topic_words if topic_words else [],
            representative_short_docs[:5],  # Use first 5 for label generation
            api_key
        )
        
        print(f"\n{'='*80}")
        print(f"Topic {topic_id}: {topic_name}")
        print(f"{'='*80}")
        
        # Print top terms for reference
        if topic_words:
            print("\nTop 10 Terms:")
            for word, weight in topic_words[:10]:
                print(f"  {word}: {weight:.4f}")
        
        
        # Build ALL arguments with short, full descriptions, and sample_id
        all_arguments = []
        for doc_idx in doc_indices:
            short_desc = docs[doc_idx]
            full_desc = short_to_full.get(short_desc, "")
            sample_id = sample_ids[doc_idx]
            all_arguments.append({
                "sample_id": sample_id,
                "short_description": short_desc,
                "full_description": full_desc
            })
        
        # Store in report data
        topic_data = {
            "name": topic_name,
            "count": int(topic_freq[topic_freq['Topic'] == topic_id].iloc[0]['Count']),
            "arguments": all_arguments
        }
        report_data["topics"].append(topic_data)
        
        # Add to summary text file
        summary_lines.append(f"Topic {topic_id}: {topic_name}")
        summary_lines.append(f"Count: {len(doc_indices)} arguments")
        summary_lines.append("-" * 80)
        summary_lines.append("Representative short descriptions (top 10):")
        for idx, doc_idx in enumerate(doc_indices[:10], 1):
            short_desc = docs[doc_idx]
            summary_lines.append(f"  {idx}. {short_desc}")
        summary_lines.append("")
        summary_lines.append("")
    
    # Save clusters JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    
    # Save summary text file
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n{'='*80}")
    print(f"JSON report saved to: {output_path}")
    print(f"Summary text saved to: {summary_path}")
    print(f"{'='*80}")
    
    return report_data


def main(input_file: str, output_file: str = None, per_sample: bool = False):
    """
    Main function to run BERTopic clustering analysis.
    
    Args:
        input_file: Path to the explanations JSON file
        output_file: Path to save the output clusters file. If None, automatically generates
                     <input_basename>_<timestamp>_clusters.json (or _per_sample_clusters.json when
                     per_sample is True) inside sample_lab/clusters/
        per_sample: If True, cluster arguments separately for each sample
    """
    # Auto-generate output file name if not provided
    if output_file is None:
        # Save to sample_lab/clusters/
        script_dir = Path(__file__).parent
        input_path = Path(input_file)
        input_basename = input_path.stem  # Get filename without extension
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = "_per_sample_clusters.json" if per_sample else "_clusters.json"
        output_file = script_dir / "clusters" / f"{input_basename}_{timestamp}{suffix}"
        print(f"Auto-generated output file: {output_file}")
    # Hyperparameters (hardcoded)
    n_neighbors = 15
    min_cluster_size = 5
    min_samples = 3
    min_df = 1
    random_state = 42
    
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    import random
    random.seed(random_state)
    
    # Load environment variables from parent directory
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Create output directory
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    mode_str = "PER-SAMPLE" if per_sample else "GLOBAL"
    print("="*80)
    print(f"BERTOPIC CLUSTERING FOR EVALUATION FEATURES ({mode_str})")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Random state: {random_state}")
    print(f"UMAP n_neighbors: {n_neighbors}")
    print(f"HDBSCAN min_cluster_size: {min_cluster_size}")
    print(f"HDBSCAN min_samples: {min_samples}")
    print(f"CountVectorizer min_df: {min_df}")
    print("="*80 + "\n")
    
    # Extract evaluation reasons from JSON
    docs, short_to_full, sample_ids = extract_eval_reasons(input_file)
    
    if not docs:
        print("Error: No documents extracted from input file")
        return
    
    total_docs = len(docs)
    print(f"\nTotal documents to cluster: {total_docs}")
    print(f"Total unique samples: {len(set(sample_ids))}")
    
    # Get embeddings from OpenAI via OpenRouter
    partial_info = None
    try:
        embeddings = get_openai_embeddings(docs, api_key)
    except EmbeddingAPIError as err:
        if not err.partial_embeddings is None and err.processed_count > 0:
            print(
                f"\nWARNING: Embedding generation failed after {err.processed_count}/{total_docs} documents.\n"
                "Proceeding with the successfully processed subset."
            )
            docs = docs[:err.processed_count]
            sample_ids = sample_ids[:err.processed_count]
            embeddings = err.partial_embeddings
            partial_info = {
                "error": str(err),
                "processed_arguments": err.processed_count,
                "total_arguments": total_docs
            }
        else:
            print("\n✗ Embedding generation failed before any documents were processed.")
            raise
    
    if per_sample:
        generate_per_sample_clusters(
            docs=docs,
            embeddings=embeddings,
            short_to_full=short_to_full,
            sample_ids=sample_ids,
            output_path=output_path,
            api_key=api_key,
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            min_df=min_df,
            random_state=random_state,
            partial_info=partial_info
        )
    else:
        # Create embedding model for BERTopic
        embedding_model = OpenRouterEmbeddings(api_key)
        
        # Create topic model
        print("\nCreating topic model with LLM-based topic labeling...")
        topic_model = create_topic_model(
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            min_df=min_df,
            random_state=random_state,
            embedding_model=embedding_model,
            api_key=api_key
        )
        
        # Fit and reduce topics
        topics, probs = fit_and_reduce_topics(topic_model, docs, embeddings)
        
        # Generate report with LLM-based topic labels
        generate_report(
            topic_model,
            docs,
            topics,
            output_path,
            short_to_full,
            sample_ids,
            api_key,
            partial_info=partial_info
        )
    
    print("\n✓ Clustering analysis complete!")


def cli_main():
    parser = argparse.ArgumentParser(
        description="Cluster evaluation-like arguments using BERTopic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the explanations JSON file (e.g., map_samples_on_leaderboard/*.json)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path for the clusters JSON file."
    )
    parser.add_argument(
        "--per-sample-clusters",
        action="store_true",
        help="Cluster each sample individually instead of clustering across all samples."
    )
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, per_sample=args.per_sample_clusters)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_main()
    else:
        print("=" * 80)
        print("RUNNING CLUSTERING")
        print("=" * 80)
        script_dir = Path(__file__).parent
        default_input = script_dir / "map_samples_on_leaderboard" / "oversight_subversion_basic_model:gpt_5_mini_leaderboard:grok_4_fast.json"
        divide_into_samples = True
        if default_input.exists():
            print(f"Using default input file: {default_input}")
            main(str(default_input), per_sample=divide_into_samples)
        else:
            print(f"Default input file not found: {default_input}")
            print("Provide an input path, e.g.:")
            print("  python cluster_eval_features.py sample_lab/map_samples_on_leaderboard/your_file.json")

