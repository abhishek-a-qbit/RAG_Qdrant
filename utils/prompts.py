from typing import Dict, Any

class PromptTemplates:
    """Collection of prompt templates for various RAG operations"""
    
    # QA Prompt Template
    QA_TEMPLATE = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and relevant to the question.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    # Document Summarization Template
    SUMMARY_TEMPLATE = """Please provide a comprehensive summary of the following document.
    Focus on the main points, key findings, and important details.
    The summary should be well-structured and easy to understand.
    
    Document: {document}
    
    Summary:"""
    
    # Keyword Extraction Template
    KEYWORD_TEMPLATE = """Extract the most important keywords and key phrases from the following text.
    Return them as a comma-separated list. Focus on terms that would be useful for search and retrieval.
    
    Text: {text}
    
    Keywords:"""
    
    # Question Generation Template
    QUESTION_GENERATION_TEMPLATE = """Based on the following context, generate {num_questions} relevant questions
    that could be asked about this content. The questions should be diverse and cover different aspects of the text.
    
    Context: {context}
    
    Generated Questions:"""
    
    # Document Classification Template
    CLASSIFICATION_TEMPLATE = """Classify the following document into one of these categories:
    {categories}
    
    Document: {document}
    
    Category:"""
    
    # Entity Extraction Template
    ENTITY_TEMPLATE = """Extract all named entities (people, organizations, locations, dates, etc.) from the following text.
    Return them in a structured format with entity type and entity name.
    
    Text: {text}
    
    Entities:"""
    
    # Sentiment Analysis Template
    SENTIMENT_TEMPLATE = """Analyze the sentiment of the following text and provide:
    1. Overall sentiment (positive, negative, neutral)
    2. Confidence score (0-1)
    3. Brief explanation
    
    Text: {text}
    
    Analysis:"""
    
    # Document Comparison Template
    COMPARISON_TEMPLATE = """Compare the following two documents and identify:
    1. Key similarities
    2. Key differences
    3. Unique insights from each document
    
    Document 1: {document1}
    
    Document 2: {document2}
    
    Comparison:"""
    
    # Code Explanation Template
    CODE_EXPLANATION_TEMPLATE = """Explain the following code in detail:
    - What the code does
    - How it works
    - Key functions and their purposes
    - Any important patterns or best practices used
    
    Code: {code}
    
    Explanation:"""
    
    # Data Analysis Template
    DATA_ANALYSIS_TEMPLATE = """Analyze the following data and provide insights:
    - Key trends and patterns
    - Statistical summary
    - Notable observations
    - Recommendations based on the data
    
    Data: {data}
    
    Analysis:"""
    
    @staticmethod
    def get_template(template_name: str) -> str:
        """Get a specific template by name"""
        templates = {
            "qa": PromptTemplates.QA_TEMPLATE,
            "summary": PromptTemplates.SUMMARY_TEMPLATE,
            "keywords": PromptTemplates.KEYWORD_TEMPLATE,
            "questions": PromptTemplates.QUESTION_GENERATION_TEMPLATE,
            "classification": PromptTemplates.CLASSIFICATION_TEMPLATE,
            "entities": PromptTemplates.ENTITY_TEMPLATE,
            "sentiment": PromptTemplates.SENTIMENT_TEMPLATE,
            "comparison": PromptTemplates.COMPARISON_TEMPLATE,
            "code": PromptTemplates.CODE_EXPLANATION_TEMPLATE,
            "analysis": PromptTemplates.DATA_ANALYSIS_TEMPLATE
        }
        return templates.get(template_name, PromptTemplates.QA_TEMPLATE)
    
    @staticmethod
    def format_template(template_name: str, **kwargs) -> str:
        """Format a template with provided variables"""
        template = PromptTemplates.get_template(template_name)
        return template.format(**kwargs)
