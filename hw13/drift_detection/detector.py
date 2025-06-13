import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DriftMetrics:
    """Container for drift detection metrics"""
    metric_name: str
    reference_value: float
    current_value: float
    drift_score: float
    threshold: float
    has_drift: bool
    confidence: float

class DriftDetector:
    """Comprehensive drift detection for book recommendation system"""
    
    def __init__(self, 
                 statistical_threshold: float = 0.05,
                 semantic_threshold: float = 0.8,
                 distribution_threshold: float = 0.1):
        """
        Initialize drift detector with configurable thresholds.
        
        Args:
            statistical_threshold: P-value threshold for statistical tests
            semantic_threshold: Cosine similarity threshold for semantic drift
            distribution_threshold: Jensen-Shannon divergence threshold
        """
        self.statistical_threshold = statistical_threshold
        self.semantic_threshold = semantic_threshold
        self.distribution_threshold = distribution_threshold
        self.logger = logging.getLogger(__name__)
        
    def detect_input_drift(self, 
                          reference_queries: List[str], 
                          current_queries: List[str]) -> Dict[str, DriftMetrics]:
        """
        Detect drift in input queries using multiple approaches:
        1. Semantic similarity (TF-IDF + cosine similarity)
        2. Query length distribution
        3. Vocabulary drift
        4. Statistical tests
        5. Language pattern drift
        """
        drift_results = {}
        
        self.logger.info(f"Starting input drift detection: {len(reference_queries)} ref vs {len(current_queries)} current")
        
        # 1. Semantic Drift Detection
        try:
            semantic_drift = self._detect_semantic_drift(reference_queries, current_queries)
            drift_results['semantic_similarity'] = semantic_drift
        except Exception as e:
            self.logger.error(f"Semantic drift detection failed: {e}")
            drift_results['semantic_similarity'] = self._create_error_metric("semantic_similarity")
        
        # 2. Query Length Distribution Drift
        try:
            length_drift = self._detect_length_distribution_drift(reference_queries, current_queries)
            drift_results['query_length_distribution'] = length_drift
        except Exception as e:
            self.logger.error(f"Length distribution drift detection failed: {e}")
            drift_results['query_length_distribution'] = self._create_error_metric("query_length_distribution")
        
        # 3. Vocabulary Drift
        try:
            vocab_drift = self._detect_vocabulary_drift(reference_queries, current_queries)
            drift_results['vocabulary_drift'] = vocab_drift
        except Exception as e:
            self.logger.error(f"Vocabulary drift detection failed: {e}")
            drift_results['vocabulary_drift'] = self._create_error_metric("vocabulary_drift")
        
        # 4. Statistical Distribution Drift (KS test)
        try:
            statistical_drift = self._detect_statistical_drift(reference_queries, current_queries)
            drift_results['statistical_distribution'] = statistical_drift
        except Exception as e:
            self.logger.error(f"Statistical drift detection failed: {e}")
            drift_results['statistical_distribution'] = self._create_error_metric("statistical_distribution")
        
        # 5. Language Pattern Drift (Ukrainian vs English content)
        try:
            language_drift = self._detect_language_pattern_drift(reference_queries, current_queries)
            drift_results['language_patterns'] = language_drift
        except Exception as e:
            self.logger.error(f"Language pattern drift detection failed: {e}")
            drift_results['language_patterns'] = self._create_error_metric("language_patterns")
        
        self.logger.info(f"Input drift detection completed: {sum(1 for m in drift_results.values() if m.has_drift)} metrics show drift")
        return drift_results
    
    def detect_output_drift(self, 
                           reference_results: List[Dict], 
                           current_results: List[Dict]) -> Dict[str, DriftMetrics]:
        """
        Detect drift in output recommendations:
        1. Genre distribution drift
        2. Results count drift
        3. Book ID distribution drift
        4. Title similarity drift
        5. Recommendation diversity drift
        """
        drift_results = {}
        
        self.logger.info(f"Starting output drift detection: {len(reference_results)} ref vs {len(current_results)} current")
        
        # 1. Genre Distribution Drift
        try:
            genre_drift = self._detect_genre_distribution_drift(reference_results, current_results)
            drift_results['genre_distribution'] = genre_drift
        except Exception as e:
            self.logger.error(f"Genre distribution drift detection failed: {e}")
            drift_results['genre_distribution'] = self._create_error_metric("genre_distribution")
        
        # 2. Results Count Drift
        try:
            count_drift = self._detect_results_count_drift(reference_results, current_results)
            drift_results['results_count'] = count_drift
        except Exception as e:
            self.logger.error(f"Results count drift detection failed: {e}")
            drift_results['results_count'] = self._create_error_metric("results_count")
        
        # 3. Book ID Distribution Drift (popularity drift)
        try:
            book_id_drift = self._detect_book_id_distribution_drift(reference_results, current_results)
            drift_results['book_popularity'] = book_id_drift
        except Exception as e:
            self.logger.error(f"Book popularity drift detection failed: {e}")
            drift_results['book_popularity'] = self._create_error_metric("book_popularity")
        
        # 4. Title Semantic Drift
        try:
            title_drift = self._detect_title_semantic_drift(reference_results, current_results)
            drift_results['title_semantics'] = title_drift
        except Exception as e:
            self.logger.error(f"Title semantic drift detection failed: {e}")
            drift_results['title_semantics'] = self._create_error_metric("title_semantics")
        
        # 5. Recommendation Diversity Drift
        try:
            diversity_drift = self._detect_recommendation_diversity_drift(reference_results, current_results)
            drift_results['recommendation_diversity'] = diversity_drift
        except Exception as e:
            self.logger.error(f"Recommendation diversity drift detection failed: {e}")
            drift_results['recommendation_diversity'] = self._create_error_metric("recommendation_diversity")
        
        self.logger.info(f"Output drift detection completed: {sum(1 for m in drift_results.values() if m.has_drift)} metrics show drift")
        return drift_results
    
    def _create_error_metric(self, metric_name: str) -> DriftMetrics:
        """Create an error metric when detection fails"""
        return DriftMetrics(
            metric_name=metric_name,
            reference_value=0.0,
            current_value=0.0,
            drift_score=1.0,
            threshold=0.5,
            has_drift=True,
            confidence=1.0
        )
    
    def _detect_semantic_drift(self, ref_queries: List[str], curr_queries: List[str]) -> DriftMetrics:
        """Detect semantic drift using TF-IDF and cosine similarity"""
        try:
            if not ref_queries or not curr_queries:
                return self._create_error_metric("semantic_similarity")
                
            # Clean and prepare queries
            ref_clean = [str(q).strip() for q in ref_queries if q and str(q).strip()]
            curr_clean = [str(q).strip() for q in curr_queries if q and str(q).strip()]
            
            if len(ref_clean) < 2 or len(curr_clean) < 2:
                self.logger.warning("Insufficient data for semantic drift detection")
                return DriftMetrics("semantic_similarity", 1.0, 0.5, 0.5, self.semantic_threshold, True, 0.8)
            
            # Combine all queries for TF-IDF fitting
            all_queries = ref_clean + curr_clean
            
            # Fit TF-IDF vectorizer with Ukrainian support
            vectorizer = TfidfVectorizer(
                max_features=1000, 
                stop_words=None,  # Keep Ukrainian words
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2)
            )
            vectorizer.fit(all_queries)
            
            # Transform reference and current queries
            ref_vectors = vectorizer.transform(ref_clean)
            curr_vectors = vectorizer.transform(curr_clean)
            
            # Calculate average vectors
            ref_avg = np.mean(ref_vectors.toarray(), axis=0)
            curr_avg = np.mean(curr_vectors.toarray(), axis=0)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([ref_avg], [curr_avg])[0][0]
            
            # Handle NaN values
            if np.isnan(similarity):
                similarity = 0.0
            
            # Drift score (1 - similarity, higher means more drift)
            drift_score = 1 - similarity
            has_drift = similarity < self.semantic_threshold
            
            return DriftMetrics(
                metric_name="semantic_similarity",
                reference_value=1.0,  # Perfect similarity baseline
                current_value=similarity,
                drift_score=drift_score,
                threshold=self.semantic_threshold,
                has_drift=has_drift,
                confidence=abs(similarity - self.semantic_threshold) / self.semantic_threshold if self.semantic_threshold > 0 else 1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in semantic drift detection: {e}")
            return self._create_error_metric("semantic_similarity")
    
    def _detect_length_distribution_drift(self, ref_queries: List[str], curr_queries: List[str]) -> DriftMetrics:
        """Detect drift in query length distribution"""
        try:
            ref_lengths = [len(str(q).split()) for q in ref_queries if q]
            curr_lengths = [len(str(q).split()) for q in curr_queries if q]
            
            if len(ref_lengths) < 3 or len(curr_lengths) < 3:
                return DriftMetrics("query_length_distribution", 5.0, 3.0, 0.4, self.statistical_threshold, False, 0.6)
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_lengths, curr_lengths)
            
            has_drift = p_value < self.statistical_threshold
            
            ref_mean = np.mean(ref_lengths)
            curr_mean = np.mean(curr_lengths)
            
            return DriftMetrics(
                metric_name="query_length_distribution",
                reference_value=ref_mean,
                current_value=curr_mean,
                drift_score=ks_stat,
                threshold=self.statistical_threshold,
                has_drift=has_drift,
                confidence=1 - p_value if p_value < 0.5 else p_value
            )
            
        except Exception as e:
            self.logger.error(f"Error in length distribution drift detection: {e}")
            return self._create_error_metric("query_length_distribution")
    
    def _detect_vocabulary_drift(self, ref_queries: List[str], curr_queries: List[str]) -> DriftMetrics:
        """Detect vocabulary drift using Jaccard similarity"""
        try:
            # Extract unique words (support Ukrainian)
            ref_text = ' '.join([str(q) for q in ref_queries if q]).lower()
            curr_text = ' '.join([str(q) for q in curr_queries if q]).lower()
            
            # Simple word tokenization
            ref_words = set(ref_text.split())
            curr_words = set(curr_text.split())
            
            # Remove very short words
            ref_words = {w for w in ref_words if len(w) > 2}
            curr_words = {w for w in curr_words if len(w) > 2}
            
            if not ref_words or not curr_words:
                return DriftMetrics("vocabulary_drift", 100, 50, 0.5, 0.7, False, 0.7)
            
            # Calculate Jaccard similarity
            intersection = len(ref_words.intersection(curr_words))
            union = len(ref_words.union(curr_words))
            jaccard_sim = intersection / union if union > 0 else 0
            
            drift_score = 1 - jaccard_sim
            has_drift = jaccard_sim < 0.7  # 70% vocabulary overlap threshold
            
            return DriftMetrics(
                metric_name="vocabulary_drift",
                reference_value=len(ref_words),
                current_value=len(curr_words),
                drift_score=drift_score,
                threshold=0.7,
                has_drift=has_drift,
                confidence=abs(jaccard_sim - 0.7) / 0.7 if jaccard_sim < 1.0 else 0.3
            )
            
        except Exception as e:
            self.logger.error(f"Error in vocabulary drift detection: {e}")
            return self._create_error_metric("vocabulary_drift")
    
    def _detect_statistical_drift(self, ref_queries: List[str], curr_queries: List[str]) -> DriftMetrics:
        """Detect statistical drift using character-level features"""
        try:
            # Extract character-level features
            ref_features = [len(str(q)) for q in ref_queries if q]  # Character length
            curr_features = [len(str(q)) for q in curr_queries if q]
            
            if len(ref_features) < 3 or len(curr_features) < 3:
                return DriftMetrics("statistical_distribution", 50, 45, 0.1, self.statistical_threshold, False, 0.5)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(ref_features, curr_features, alternative='two-sided')
            
            has_drift = p_value < self.statistical_threshold
            
            ref_median = np.median(ref_features)
            curr_median = np.median(curr_features)
            
            return DriftMetrics(
                metric_name="statistical_distribution",
                reference_value=ref_median,
                current_value=curr_median,
                drift_score=u_stat / (len(ref_features) * len(curr_features)) if len(ref_features) * len(curr_features) > 0 else 0,
                threshold=self.statistical_threshold,
                has_drift=has_drift,
                confidence=1 - p_value if p_value < 0.5 else p_value
            )
            
        except Exception as e:
            self.logger.error(f"Error in statistical drift detection: {e}")
            return self._create_error_metric("statistical_distribution")
    
    def _detect_language_pattern_drift(self, ref_queries: List[str], curr_queries: List[str]) -> DriftMetrics:
        """Detect drift in language patterns (Ukrainian vs English content)"""
        try:
            def estimate_ukrainian_ratio(queries):
                """Estimate ratio of Ukrainian content based on Cyrillic characters"""
                if not queries:
                    return 0.0
                    
                total_chars = 0
                cyrillic_chars = 0
                
                for query in queries:
                    if not query:
                        continue
                    query_str = str(query)
                    total_chars += len(query_str)
                    cyrillic_chars += sum(1 for c in query_str if '\u0400' <= c <= '\u04FF')
                
                return cyrillic_chars / total_chars if total_chars > 0 else 0.0
            
            ref_ua_ratio = estimate_ukrainian_ratio(ref_queries)
            curr_ua_ratio = estimate_ukrainian_ratio(curr_queries)
            
            # Calculate drift based on language pattern change
            ratio_diff = abs(ref_ua_ratio - curr_ua_ratio)
            drift_score = ratio_diff
            has_drift = ratio_diff > 0.2  # 20% change in language pattern
            
            return DriftMetrics(
                metric_name="language_patterns",
                reference_value=ref_ua_ratio,
                current_value=curr_ua_ratio,
                drift_score=drift_score,
                threshold=0.2,
                has_drift=has_drift,
                confidence=ratio_diff / 0.2 if ratio_diff < 0.2 else 1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in language pattern drift detection: {e}")
            return self._create_error_metric("language_patterns")
    
    def _detect_genre_distribution_drift(self, ref_results: List[Dict], curr_results: List[Dict]) -> DriftMetrics:
        """Detect drift in genre distribution of recommendations"""
        try:
            # Extract genres from results
            ref_genres = []
            curr_genres = []
            
            for result_list in ref_results:
                if isinstance(result_list, list):
                    ref_genres.extend([r.get('genre', 'unknown') for r in result_list if isinstance(r, dict)])
                elif isinstance(result_list, dict):
                    ref_genres.append(result_list.get('genre', 'unknown'))
            
            for result_list in curr_results:
                if isinstance(result_list, list):
                    curr_genres.extend([r.get('genre', 'unknown') for r in result_list if isinstance(r, dict)])
                elif isinstance(result_list, dict):
                    curr_genres.append(result_list.get('genre', 'unknown'))
            
            if not ref_genres and not curr_genres:
                return DriftMetrics("genre_distribution", 0, 0, 0, self.distribution_threshold, False, 0.5)
            
            # Handle case where one list is empty
            if not ref_genres:
                ref_genres = ['unknown']
            if not curr_genres:
                curr_genres = ['unknown']
            
            # Calculate genre distributions
            ref_genre_counts = pd.Series(ref_genres).value_counts(normalize=True)
            curr_genre_counts = pd.Series(curr_genres).value_counts(normalize=True)
            
            # Align distributions (fill missing genres with 0)
            all_genres = set(ref_genre_counts.index) | set(curr_genre_counts.index)
            ref_dist = [ref_genre_counts.get(g, 0) for g in all_genres]
            curr_dist = [curr_genre_counts.get(g, 0) for g in all_genres]
            
            # Jensen-Shannon divergence
            js_distance = self._jensen_shannon_distance(ref_dist, curr_dist)
            
            has_drift = js_distance > self.distribution_threshold
            
            return DriftMetrics(
                metric_name="genre_distribution",
                reference_value=len(ref_genre_counts),
                current_value=len(curr_genre_counts),
                drift_score=js_distance,
                threshold=self.distribution_threshold,
                has_drift=has_drift,
                confidence=js_distance / self.distribution_threshold if js_distance < self.distribution_threshold else 1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in genre distribution drift detection: {e}")
            return self._create_error_metric("genre_distribution")
    
    def _detect_results_count_drift(self, ref_results: List[Dict], curr_results: List[Dict]) -> DriftMetrics:
        """Detect drift in number of results returned"""
        try:
            ref_counts = []
            curr_counts = []
            
            for r in ref_results:
                if isinstance(r, list):
                    ref_counts.append(len(r))
                elif isinstance(r, dict):
                    ref_counts.append(1)
                else:
                    ref_counts.append(0)
                    
            for r in curr_results:
                if isinstance(r, list):
                    curr_counts.append(len(r))
                elif isinstance(r, dict):
                    curr_counts.append(1)
                else:
                    curr_counts.append(0)
            
            if len(ref_counts) < 3 or len(curr_counts) < 3:
                ref_mean = np.mean(ref_counts) if ref_counts else 0
                curr_mean = np.mean(curr_counts) if curr_counts else 0
                return DriftMetrics("results_count", ref_mean, curr_mean, 0.1, self.statistical_threshold, False, 0.5)
            
            # Statistical test
            u_stat, p_value = stats.mannwhitneyu(ref_counts, curr_counts, alternative='two-sided')
            
            has_drift = p_value < self.statistical_threshold
            
            ref_mean = np.mean(ref_counts)
            curr_mean = np.mean(curr_counts)
            
            return DriftMetrics(
                metric_name="results_count",
                reference_value=ref_mean,
                current_value=curr_mean,
                drift_score=abs(ref_mean - curr_mean) / max(ref_mean, 1),
                threshold=self.statistical_threshold,
                has_drift=has_drift,
                confidence=1 - p_value if p_value < 0.5 else p_value
            )
            
        except Exception as e:
            self.logger.error(f"Error in results count drift detection: {e}")
            return self._create_error_metric("results_count")
    
    def _detect_book_id_distribution_drift(self, ref_results: List[Dict], curr_results: List[Dict]) -> DriftMetrics:
        """Detect drift in book popularity (which books are being recommended)"""
        try:
            # Extract book IDs
            ref_book_ids = []
            curr_book_ids = []
            
            for result_list in ref_results:
                if isinstance(result_list, list):
                    ref_book_ids.extend([r.get('id', 0) for r in result_list if isinstance(r, dict)])
                elif isinstance(result_list, dict):
                    ref_book_ids.append(result_list.get('id', 0))
            
            for result_list in curr_results:
                if isinstance(result_list, list):
                    curr_book_ids.extend([r.get('id', 0) for r in result_list if isinstance(r, dict)])
                elif isinstance(result_list, dict):
                    curr_book_ids.append(result_list.get('id', 0))
            
            if not ref_book_ids and not curr_book_ids:
                return DriftMetrics("book_popularity", 0, 0, 0, 0.6, False, 0.5)
            
            # Handle empty cases
            if not ref_book_ids:
                ref_book_ids = [0]
            if not curr_book_ids:
                curr_book_ids = [0]
            
            # Calculate popularity distributions
            ref_popularity = pd.Series(ref_book_ids).value_counts(normalize=True)
            curr_popularity = pd.Series(curr_book_ids).value_counts(normalize=True)
            
            # Calculate overlap in top books
            top_n = min(10, len(ref_popularity), len(curr_popularity))
            ref_top_books = set(ref_popularity.head(top_n).index)
            curr_top_books = set(curr_popularity.head(top_n).index)
            
            if not ref_top_books or not curr_top_books:
                overlap_ratio = 0.5
            else:
                overlap = len(ref_top_books.intersection(curr_top_books))
                max_overlap = max(len(ref_top_books), len(curr_top_books))
                overlap_ratio = overlap / max_overlap if max_overlap > 0 else 0
            
            drift_score = 1 - overlap_ratio
            has_drift = overlap_ratio < 0.6  # 60% overlap threshold
            
            return DriftMetrics(
                metric_name="book_popularity",
                reference_value=len(ref_popularity),
                current_value=len(curr_popularity),
                drift_score=drift_score,
                threshold=0.6,
                has_drift=has_drift,
                confidence=abs(overlap_ratio - 0.6) / 0.6 if overlap_ratio < 1.0 else 0.4
            )
            
        except Exception as e:
            self.logger.error(f"Error in book ID distribution drift detection: {e}")
            return self._create_error_metric("book_popularity")
    
    def _detect_title_semantic_drift(self, ref_results: List[Dict], curr_results: List[Dict]) -> DriftMetrics:
        """Detect semantic drift in recommended book titles"""
        try:
            # Extract titles
            ref_titles = []
            curr_titles = []
            
            for result_list in ref_results:
                if isinstance(result_list, list):
                    ref_titles.extend([r.get('title', '') for r in result_list if isinstance(r, dict)])
                elif isinstance(result_list, dict):
                    ref_titles.append(result_list.get('title', ''))
            
            for result_list in curr_results:
                if isinstance(result_list, list):
                    curr_titles.extend([r.get('title', '') for r in result_list if isinstance(r, dict)])
                elif isinstance(result_list, dict):
                    curr_titles.append(result_list.get('title', ''))
            
            # Clean titles
            ref_titles = [str(t).strip() for t in ref_titles if t and str(t).strip()]
            curr_titles = [str(t).strip() for t in curr_titles if t and str(t).strip()]
            
            # Use TF-IDF for semantic comparison
            if len(ref_titles) > 0 and len(curr_titles) > 0:
                all_titles = ref_titles + curr_titles
                
                if len(all_titles) < 3:
                    return DriftMetrics("title_semantics", 1.0, 0.8, 0.2, 0.8, False, 0.6)
                
                vectorizer = TfidfVectorizer(max_features=500, stop_words=None, min_df=1)
                vectorizer.fit(all_titles)
                
                ref_vectors = vectorizer.transform(ref_titles)
                curr_vectors = vectorizer.transform(curr_titles)
                
                ref_avg = np.mean(ref_vectors.toarray(), axis=0)
                curr_avg = np.mean(curr_vectors.toarray(), axis=0)
                
                similarity = cosine_similarity([ref_avg], [curr_avg])[0][0]
                
                if np.isnan(similarity):
                    similarity = 0.8  # Default similarity
                
                drift_score = 1 - similarity
                has_drift = similarity < 0.8
                
                return DriftMetrics(
                    metric_name="title_semantics",
                    reference_value=1.0,
                    current_value=similarity,
                    drift_score=drift_score,
                    threshold=0.8,
                    has_drift=has_drift,
                    confidence=abs(similarity - 0.8) / 0.8 if similarity < 1.0 else 0.2
                )
            else:
                return DriftMetrics("title_semantics", 0, 0, 1, 0.8, True, 1.0)
                
        except Exception as e:
            self.logger.error(f"Error in title semantic drift detection: {e}")
            return self._create_error_metric("title_semantics")
    
    def _detect_recommendation_diversity_drift(self, ref_results: List[Dict], curr_results: List[Dict]) -> DriftMetrics:
        """Detect drift in recommendation diversity (how varied the recommendations are)"""
        try:
            def calculate_diversity(results):
                """Calculate diversity score based on unique genres and books"""
                all_items = []
                for result_list in results:
                    if isinstance(result_list, list):
                        all_items.extend(result_list)
                    elif isinstance(result_list, dict):
                        all_items.append(result_list)
                
                if not all_items:
                    return 0.0
                
                # Unique genres
                genres = set(item.get('genre', 'unknown') for item in all_items if isinstance(item, dict))
                # Unique books
                books = set(item.get('id', 0) for item in all_items if isinstance(item, dict))
                
                # Diversity score combines genre and book diversity
                genre_diversity = len(genres) / max(len(all_items), 1)
                book_diversity = len(books) / max(len(all_items), 1)
                
                return (genre_diversity + book_diversity) / 2
            
            ref_diversity = calculate_diversity(ref_results)
            curr_diversity = calculate_diversity(curr_results)
            
            # Calculate drift in diversity
            diversity_diff = abs(ref_diversity - curr_diversity)
            drift_score = diversity_diff
            has_drift = diversity_diff > 0.2  # 20% change in diversity
            
            return DriftMetrics(
                metric_name="recommendation_diversity",
                reference_value=ref_diversity,
                current_value=curr_diversity,
                drift_score=drift_score,
                threshold=0.2,
                has_drift=has_drift,
                confidence=diversity_diff / 0.2 if diversity_diff < 0.2 else 1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in recommendation diversity drift detection: {e}")
            return self._create_error_metric("recommendation_diversity")
    
    def _jensen_shannon_distance(self, p: List[float], q: List[float]) -> float:
        """Calculate Jensen-Shannon distance between two probability distributions"""
        try:
            p = np.array(p) + 1e-10  # Add small epsilon to avoid log(0)
            q = np.array(q) + 1e-10
            
            # Normalize
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # Calculate JS divergence
            m = 0.5 * (p + q)
            js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
            
            # Convert to distance (square root of divergence)
            return np.sqrt(js_div) if not np.isnan(js_div) else 1.0
            
        except Exception:
            return 1.0  # Maximum distance on error

class DriftReporter:
    """Generate comprehensive drift reports with actionable insights"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_summary_report(self, 
                               input_drift: Dict[str, DriftMetrics], 
                               output_drift: Dict[str, DriftMetrics],
                               timestamp: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive drift summary report with Ukrainian-specific insights"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate overall drift scores
        input_drift_count = sum(1 for m in input_drift.values() if m.has_drift)
        output_drift_count = sum(1 for m in output_drift.values() if m.has_drift)
        
        total_metrics = len(input_drift) + len(output_drift)
        total_drift_count = input_drift_count + output_drift_count
        
        overall_drift_ratio = total_drift_count / total_metrics if total_metrics > 0 else 0
        
        # Determine alert level with enhanced logic
        alert_level = self._determine_alert_level(overall_drift_ratio, input_drift, output_drift)
        
        # Create detailed report
        report = {
            "timestamp": timestamp.isoformat(),
            "overall_summary": {
                "total_metrics_checked": total_metrics,
                "metrics_with_drift": total_drift_count,
                "drift_ratio": overall_drift_ratio,
                "alert_level": alert_level,
                "status": "DRIFT_DETECTED" if total_drift_count > 0 else "NO_DRIFT",
                "system_health": self._assess_system_health(input_drift, output_drift)
            },
            "input_drift_analysis": {
                "metrics_checked": len(input_drift),
                "drift_detected": input_drift_count,
                "critical_drifts": self._identify_critical_input_drifts(input_drift),
                "details": {name: self._metric_to_dict(metric) for name, metric in input_drift.items()}
            },
            "output_drift_analysis": {
                "metrics_checked": len(output_drift),
                "drift_detected": output_drift_count,
                "critical_drifts": self._identify_critical_output_drifts(output_drift),
                "details": {name: self._metric_to_dict(metric) for name, metric in output_drift.items()}
            },
            "detailed_insights": self._generate_detailed_insights(input_drift, output_drift),
            "recommendations": self._generate_recommendations(input_drift, output_drift),
            "next_check_recommended": (timestamp + timedelta(hours=24)).isoformat(),
            "monitoring_suggestions": self._generate_monitoring_suggestions(input_drift, output_drift)
        }
        
        return report
    
    def _determine_alert_level(self, overall_ratio: float, input_drift: Dict, output_drift: Dict) -> str:
        """Determine alert level based on drift severity and impact"""
        
        # Check for critical drifts that indicate serious system issues
        critical_metrics = ['semantic_similarity', 'genre_distribution', 'statistical_distribution', 'language_patterns']
        critical_drift = any(
            metric.has_drift and metric.metric_name in critical_metrics and metric.drift_score > 0.7
            for metric in list(input_drift.values()) + list(output_drift.values())
        )
        
        # Check for high-confidence drifts
        high_confidence_drifts = sum(
            1 for metric in list(input_drift.values()) + list(output_drift.values())
            if metric.has_drift and metric.confidence > 0.8
        )
        
        if critical_drift:
            return "CRITICAL"
        elif overall_ratio > 0.6 or high_confidence_drifts > 3:
            return "HIGH"
        elif overall_ratio > 0.3 or high_confidence_drifts > 1:
            return "MEDIUM"
        elif overall_ratio > 0:
            return "LOW"
        else:
            return "NONE"
    
    def _assess_system_health(self, input_drift: Dict, output_drift: Dict) -> str:
        """Assess overall system health based on drift patterns"""
        total_metrics = len(input_drift) + len(output_drift)
        drift_metrics = sum(1 for m in list(input_drift.values()) + list(output_drift.values()) if m.has_drift)
        
        if drift_metrics == 0:
            return "EXCELLENT"
        elif drift_metrics / total_metrics < 0.2:
            return "GOOD"
        elif drift_metrics / total_metrics < 0.4:
            return "FAIR"
        elif drift_metrics / total_metrics < 0.6:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _identify_critical_input_drifts(self, input_drift: Dict) -> List[str]:
        """Identify critical input drifts that need immediate attention"""
        critical_drifts = []
        
        for name, metric in input_drift.items():
            if metric.has_drift and metric.confidence > 0.8:
                if name == 'semantic_similarity' and metric.current_value < 0.6:
                    critical_drifts.append("Severe semantic drift detected - user query patterns have fundamentally changed")
                elif name == 'language_patterns' and metric.drift_score > 0.5:
                    critical_drifts.append("Major language pattern shift - Ukrainian/English content ratio changed significantly")
                elif name == 'vocabulary_drift' and metric.drift_score > 0.6:
                    critical_drifts.append("Extensive vocabulary drift - many new terms in user queries")
        
        return critical_drifts
    
    def _identify_critical_output_drifts(self, output_drift: Dict) -> List[str]:
        """Identify critical output drifts that need immediate attention"""
        critical_drifts = []
        
        for name, metric in output_drift.items():
            if metric.has_drift and metric.confidence > 0.8:
                if name == 'genre_distribution' and metric.drift_score > 0.3:
                    critical_drifts.append("Major genre distribution shift - recommendation patterns changed significantly")
                elif name == 'recommendation_diversity' and metric.drift_score > 0.4:
                    critical_drifts.append("Recommendation diversity drift - system may be showing bias or reduced variety")
                elif name == 'results_count' and metric.drift_score > 0.5:
                    critical_drifts.append("Results count anomaly - system returning different number of recommendations")
        
        return critical_drifts
    
    def _generate_detailed_insights(self, input_drift: Dict, output_drift: Dict) -> Dict[str, Any]:
        """Generate detailed insights about drift patterns"""
        insights = {
            "user_behavior_changes": [],
            "system_performance_issues": [],
            "recommendation_quality_concerns": [],
            "data_quality_issues": []
        }
        
        # Analyze input drift patterns
        for name, metric in input_drift.items():
            if metric.has_drift:
                if name == 'semantic_similarity':
                    if metric.current_value < 0.7:
                        insights["user_behavior_changes"].append(
                            f"Users are searching for different types of content (similarity: {metric.current_value:.2f})"
                        )
                elif name == 'language_patterns':
                    insights["user_behavior_changes"].append(
                        f"Language preference shift detected (Ukrainian ratio changed by {metric.drift_score:.2f})"
                    )
                elif name == 'query_length_distribution':
                    insights["user_behavior_changes"].append(
                        f"Query complexity changed (avg length: {metric.reference_value:.1f} → {metric.current_value:.1f})"
                    )
        
        # Analyze output drift patterns
        for name, metric in output_drift.items():
            if metric.has_drift:
                if name == 'genre_distribution':
                    insights["recommendation_quality_concerns"].append(
                        f"Genre recommendation bias detected (JS distance: {metric.drift_score:.3f})"
                    )
                elif name == 'recommendation_diversity':
                    insights["recommendation_quality_concerns"].append(
                        f"Recommendation diversity changed ({metric.reference_value:.2f} → {metric.current_value:.2f})"
                    )
                elif name == 'results_count':
                    insights["system_performance_issues"].append(
                        f"Results count inconsistency (avg: {metric.reference_value:.1f} → {metric.current_value:.1f})"
                    )
        
        return insights
    
    def _metric_to_dict(self, metric: DriftMetrics) -> Dict[str, Any]:
        """Convert DriftMetrics to dictionary with enhanced information"""
        return {
            "has_drift": metric.has_drift,
            "drift_score": round(metric.drift_score, 4),
            "threshold": metric.threshold,
            "reference_value": round(metric.reference_value, 4),
            "current_value": round(metric.current_value, 4),
            "confidence": round(metric.confidence, 4),
            "severity": self._calculate_severity(metric),
            "impact": self._assess_metric_impact(metric)
        }
    
    def _calculate_severity(self, metric: DriftMetrics) -> str:
        """Calculate severity level for a drift metric"""
        if not metric.has_drift:
            return "NONE"
        elif metric.confidence > 0.9 and metric.drift_score > 0.7:
            return "CRITICAL"
        elif metric.confidence > 0.7 and metric.drift_score > 0.5:
            return "HIGH"
        elif metric.confidence > 0.5 and metric.drift_score > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_metric_impact(self, metric: DriftMetrics) -> str:
        """Assess the potential impact of a drift metric"""
        impact_mapping = {
            'semantic_similarity': 'User Experience',
            'language_patterns': 'Localization',
            'genre_distribution': 'Recommendation Quality',
            'recommendation_diversity': 'Content Variety',
            'vocabulary_drift': 'Search Relevance',
            'results_count': 'System Performance'
        }
        
        return impact_mapping.get(metric.metric_name, 'System Health')
    
    def _generate_recommendations(self, input_drift: Dict, output_drift: Dict) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        
        # Input drift recommendations
        for name, metric in input_drift.items():
            if metric.has_drift and metric.confidence > 0.6:
                if name == "semantic_similarity":
                    recommendations.append(
                        "🔍 Semantic drift detected: Consider retraining the model with recent Ukrainian queries to improve relevance."
                    )
                elif name == "query_length_distribution":
                    recommendations.append(
                        "📏 Query pattern change: Review query preprocessing and consider adjusting search algorithms for new query lengths."
                    )
                elif name == "vocabulary_drift":
                    recommendations.append(
                        "📝 New vocabulary detected: Update preprocessing pipelines and expand Ukrainian language support."
                    )
                elif name == "language_patterns":
                    recommendations.append(
                        "🌐 Language pattern shift: Review multilingual support and consider separate models for Ukrainian/English."
                    )
                elif name == "statistical_distribution":
                    recommendations.append(
                        "📊 Statistical drift: Investigate data quality and consider updating feature engineering approaches."
                    )
        
        # Output drift recommendations
        for name, metric in output_drift.items():
            if metric.has_drift and metric.confidence > 0.6:
                if name == "genre_distribution":
                    recommendations.append(
                        "📚 Genre bias detected: Review recommendation algorithm for genre balancing and user preference shifts."
                    )
                elif name == "book_popularity":
                    recommendations.append(
                        "⭐ Popularity shift: Monitor for recommendation bias and consider diversification strategies."
                    )
                elif name == "results_count":
                    recommendations.append(
                        "🔢 Results inconsistency: Check search filtering logic and ranking algorithms for stability."
                    )
                elif name == "title_semantics":
                    recommendations.append(
                        "📖 Title semantic drift: Review content similarity calculations and update semantic models."
                    )
                elif name == "recommendation_diversity":
                    recommendations.append(
                        "🎯 Diversity change: Implement diversity boosting techniques and monitor user satisfaction."
                    )
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ No significant drift detected. System performing within expected parameters.")
        else:
            recommendations.insert(0, "🚨 Priority actions needed based on drift detection results:")
            recommendations.append("📈 Consider implementing A/B testing to validate changes before full deployment.")
            recommendations.append("🔄 Schedule more frequent monitoring during periods of detected drift.")
        
        return recommendations
    
    def _generate_monitoring_suggestions(self, input_drift: Dict, output_drift: Dict) -> List[str]:
        """Generate suggestions for enhanced monitoring"""
        suggestions = []
        
        total_drifts = sum(1 for m in list(input_drift.values()) + list(output_drift.values()) if m.has_drift)
        
        if total_drifts > 3:
            suggestions.append("Increase monitoring frequency to hourly during drift period")
            suggestions.append("Set up real-time alerts for critical metrics")
        elif total_drifts > 0:
            suggestions.append("Monitor daily for the next week")
            suggestions.append("Track user satisfaction metrics closely")
        
        suggestions.extend([
            "Collect user feedback on recommendation quality",
            "Monitor business KPIs (engagement, conversion)",
            "Track model performance metrics",
            "Review data quality indicators"
        ])
        
        return suggestions