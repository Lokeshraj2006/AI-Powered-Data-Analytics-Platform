"""
Enhanced Gemini DataFrame Agent - Core functionality for natural language data analysis
"""

import pandas as pd
import numpy as np
import logging
import time
import asyncio
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime

# LangChain imports
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType

# Local imports
from ..config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

class GeminiDataFrameAgent:
    """
    Advanced DataFrame Agent powered by Gemini API
    Supports natural language queries on pandas DataFrames
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        enable_memory: bool = True,
        max_iterations: int = 5
    ):
        """Initialize the Gemini DataFrame Agent"""
        
        self.api_key = api_key or settings.gemini_api_key
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Initialize LangChain LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
        
        # Memory for conversation history
        if enable_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        else:
            self.memory = None
        
        # Initialize data storage
        self.dataframes = {}
        self.current_df = None
        self.agent = None
        
        # Session statistics
        self.session_stats = {
            "queries_executed": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "dataframes_loaded": 0,
            "session_start": datetime.now()
        }
        
        logger.info(f"🤖 GeminiDataFrameAgent initialized with model: {self.model}")
    
    def load_dataframe(
        self, 
        data: Union[str, pd.DataFrame, dict], 
        name: str = "main_df",
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Load and preprocess a DataFrame from various sources
        
        Args:
            data: File path, DataFrame, or dict
            name: Name for the DataFrame
            preprocess: Whether to apply smart preprocessing
        
        Returns:
            Dict with loading status and metadata
        """
        start_time = time.time()
        
        try:
            # Load data based on type
            if isinstance(data, str):
                df = self._load_from_file(data)
                source = f"file: {data}"
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
                source = "pandas DataFrame"
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
                source = "dictionary"
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Apply preprocessing if requested
            if preprocess:
                df = self._preprocess_dataframe(df)
            
            # Store the DataFrame
            self.dataframes[name] = df
            self.current_df = df
            
            # Create LangChain agent for this DataFrame
            self._create_agent(df)
            
            # Update statistics
            self.session_stats["dataframes_loaded"] += 1
            load_time = time.time() - start_time
            
            # Generate data summary
            summary = self._generate_data_summary(df)
            
            result = {
                "success": True,
                "message": f"✅ Successfully loaded DataFrame '{name}'",
                "dataframe_name": name,
                "source": source,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "load_time": round(load_time, 3),
                "summary": summary,
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
            
            logger.info(f"📊 DataFrame '{name}' loaded: {df.shape} ({load_time:.2f}s)")
            return result
            
        except Exception as e:
            error_msg = f"❌ Failed to load DataFrame '{name}': {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": str(e),
                "message": error_msg,
                "dataframe_name": name
            }
    
    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load DataFrame from file with automatic format detection"""
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to read {file_extension} file: {str(e)}")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent preprocessing to the DataFrame"""
        
        logger.info("🔄 Applying intelligent preprocessing...")
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # 1. Convert date columns
        date_columns = []
        for col in processed_df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                try:
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='ignore')
                    if processed_df[col].dtype == 'datetime64[ns]':
                        date_columns.append(col)
                except:
                    pass
        
        # 2. Remove duplicate rows
        initial_rows = len(processed_df)
        processed_df = processed_df.drop_duplicates()
        removed_duplicates = initial_rows - len(processed_df)
        
        if removed_duplicates > 0:
            logger.info(f"📊 Removed {removed_duplicates} duplicate rows")
        
        if date_columns:
            logger.info(f"📅 Converted date columns: {date_columns}")
        
        logger.info(f"✅ Preprocessing completed: {processed_df.shape}")
        
        return processed_df
    
    def _create_agent(self, df: pd.DataFrame):
        """Create LangChain pandas DataFrame agent"""
        
        try:
            self.agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=settings.debug,
                return_intermediate_steps=True,
                max_iterations=self.max_iterations,
                allow_dangerous_code=True,  # Required for pandas operations
                include_df_in_prompt=True,
                number_of_head_rows=5
            )
            
            logger.info("🔗 LangChain DataFrame agent created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create agent: {e}")
            raise
    
    def _generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive summary of the DataFrame"""
        
        summary = {
            "shape": df.shape,
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            "missing_values": df.isnull().sum().sum(),
            "missing_percentage": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
        }
        
        return summary
    
    async def query(
        self, 
        question: str, 
        use_cache: bool = True,
        explain_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Process natural language queries about the DataFrame
        
        Args:
            question: Natural language question
            use_cache: Whether to use cached responses
            explain_reasoning: Whether to include reasoning steps
            
        Returns:
            Dict with query results and metadata
        """
        
        if self.agent is None or self.current_df is None:
            return {
                "success": False,
                "error": "No DataFrame loaded. Please load a DataFrame first.",
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
        
        start_time = time.time()
        self.session_stats["queries_executed"] += 1
        
        try:
            # Execute the query using LangChain agent
            logger.info(f"🔍 Processing query: {question[:100]}...")
            
            # Run the agent
            agent_result = self.agent({"input": question})
            
            # Process the response
            response_time = time.time() - start_time
            self.session_stats["successful_queries"] += 1
            self.session_stats["total_response_time"] += response_time
            self.session_stats["avg_response_time"] = (
                self.session_stats["total_response_time"] / self.session_stats["successful_queries"]
            )
            
            result = {
                "success": True,
                "question": question,
                "answer": agent_result["output"],
                "response_time": round(response_time, 3),
                "timestamp": datetime.now().isoformat(),
                "intermediate_steps": agent_result.get("intermediate_steps", []) if explain_reasoning else None,
                "suggestions": self._generate_followup_suggestions(question, agent_result["output"])
            }
            
            logger.info(f"✅ Query completed successfully ({response_time:.2f}s)")
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.session_stats["failed_queries"] += 1
            
            error_msg = str(e)
            logger.error(f"❌ Query failed: {error_msg}")
            
            return {
                "success": False,
                "question": question,
                "error": error_msg,
                "error_type": type(e).__name__,
                "response_time": round(response_time, 3),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Try rephrasing your question more clearly",
                    "Check if column names are spelled correctly",
                    "Make sure your question relates to the data available",
                    "Try asking about basic statistics first (mean, count, etc.)"
                ]
            }
    
    def _generate_followup_suggestions(self, question: str, answer: str) -> List[str]:
        """Generate intelligent follow-up question suggestions"""
        
        suggestions = []
        question_lower = question.lower()
        
        # Analyze the question type and suggest related queries
        if any(word in question_lower for word in ["average", "mean"]):
            suggestions.extend([
                "What's the median and mode for this data?",
                "Show me the distribution of this variable",
                "Are there any outliers in this data?"
            ])
        
        if any(word in question_lower for word in ["count", "how many"]):
            suggestions.extend([
                "What's the percentage breakdown?",
                "Show me the top 10 categories",
                "Compare this with other columns"
            ])
        
        # Generic suggestions
        if not suggestions:
            suggestions = [
                "Show me summary statistics for all columns",
                "What are the unique values in categorical columns?",
                "Are there any missing values in the data?",
                "Show me the data types of all columns"
            ]
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        
        return {
            **self.session_stats,
            "success_rate": round(
                (self.session_stats["successful_queries"] / max(self.session_stats["queries_executed"], 1)) * 100, 2
            ),
            "loaded_dataframes": list(self.dataframes.keys()),
            "current_dataframe_shape": self.current_df.shape if self.current_df is not None else None,
            "uptime_minutes": round((datetime.now() - self.session_stats["session_start"]).total_seconds() / 60, 1)
        }
    
    def clear_cache(self):
        """Clear query cache"""
        logger.info("🗑️ Cache cleared")

# Convenience function for quick initialization
def create_agent(
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.1
) -> GeminiDataFrameAgent:
    """Create a new GeminiDataFrameAgent instance"""
    return GeminiDataFrameAgent(
        api_key=api_key,
        model=model,
        temperature=temperature
    )