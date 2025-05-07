import os
import logging
from typing import Dict, List, Any

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Templates for different course content types
MODULE_TEMPLATE = """
## {module_title}

{module_content}
"""

SOFTWARE_MODULE_TEMPLATE = """
## {module_title}

{module_content}

### Code Examples

{code_examples}
"""

COURSE_TEMPLATE = """
# {course_title}

{course_description}

## Table of Contents
{table_of_contents}

{modules_content}
"""

def generate_course_content(course_data: Dict[str, Any]) -> str:
    """
    Generate complete course content based on the provided course structure.
    
    Args:
        course_data (Dict): Course data including title and modules
        
    Returns:
        str: Complete course content in markdown format
    """
    try:
        logger.info(f"Generating content for course: {course_data['title']}")
        
        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=0.7, 
            model_name="gpt-4",
            api_key=OPENAI_API_KEY
        )
        
        # Generate course description
        course_description = generate_course_description(llm, course_data)
        
        # Generate table of contents
        toc = generate_table_of_contents(course_data['modules'])
        
        # Generate content for each module
        modules_content = []
        for module in course_data['modules']:
            module_content = generate_module_content(llm, module, is_software_course('software' in course_data.get('category', '').lower() or 'programming' in course_data.get('category', '').lower()))
            modules_content.append(module_content)
        
        # Combine everything into a complete course
        complete_course = COURSE_TEMPLATE.format(
            course_title=course_data['title'],
            course_description=course_description,
            table_of_contents=toc,
            modules_content="\n\n".join(modules_content)
        )
        
        return complete_course
        
    except Exception as e:
        logger.error(f"Error generating course content: {e}")
        raise


def generate_course_description(llm, course_data: Dict[str, Any]) -> str:
    """
    Generate a comprehensive course description.
    
    Args:
        llm: Language model instance
        course_data (Dict): Course data
        
    Returns:
        str: Course description
    """
    prompt_template = PromptTemplate(
        input_variables=["title", "modules"],
        template="""
        Create a comprehensive and engaging course description for a course titled "{title}".
        The course contains the following modules: {modules}.
        
        The description should:
        1. Explain what students will learn
        2. Highlight the importance of this knowledge
        3. Mention the practical applications
        4. Be engaging and motivating
        5. Be between 150-200 words
        
        Description:
        """
    )
    
    # Extract module titles for the prompt
    module_titles = [module['title'] for module in course_data['modules']]
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain
    description = chain.run(title=course_data['title'], modules=", ".join(module_titles))
    
    return description.strip()


def generate_table_of_contents(modules: List[Dict[str, Any]]) -> str:
    """
    Generate a table of contents from the modules.
    
    Args:
        modules (List): List of module dictionaries
        
    Returns:
        str: Table of contents in markdown format
    """
    toc = []
    
    for i, module in enumerate(modules, 1):
        module_entry = f"{i}. [{module['title']}](#{module['title'].lower().replace(' ', '-')})"
        toc.append(module_entry)
        
        # Add submodules/sections if present
        if 'sections' in module:
            for j, section in enumerate(module['sections'], 1):
                section_entry = f"   {i}.{j}. [{section['title']}](#{section['title'].lower().replace(' ', '-')})"
                toc.append(section_entry)
    
    return "\n".join(toc)


def generate_module_content(llm, module: Dict[str, Any], is_software: bool) -> str:
    """
    Generate content for a single module.
    
    Args:
        llm: Language model instance
        module (Dict): Module data
        is_software (bool): Whether this is a software/programming course
        
    Returns:
        str: Module content in markdown format
    """
    # Generate content for this module
    if is_software:
        return generate_software_module(llm, module)
    else:
        return generate_standard_module(llm, module)


def generate_standard_module(llm, module: Dict[str, Any]) -> str:
    """
    Generate content for a standard (non-software) module.
    
    Args:
        llm: Language model instance
        module (Dict): Module data
        
    Returns:
        str: Module content in markdown format
    """
    prompt_template = PromptTemplate(
        input_variables=["title", "sections"],
        template="""
        Create detailed educational content for a module titled "{title}" that covers the following sections: {sections}.
        
        For each section:
        1. Provide a clear explanation of key concepts
        2. Include relevant examples
        3. Add practical applications where appropriate
        4. Format the content with proper markdown headings, bullet points, and formatting
        5. Make sure the content is informative, accurate, and educational
        
        Return the complete module content in rich markdown format:
        """
    )
    
    # Extract section titles for the prompt
    section_titles = []
    if 'sections' in module:
        section_titles = [section['title'] for section in module['sections']]
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain
    content = chain.run(title=module['title'], sections=", ".join(section_titles) if section_titles else "No specific sections")
    
    return MODULE_TEMPLATE.format(
        module_title=module['title'],
        module_content=content.strip()
    )


def generate_software_module(llm, module: Dict[str, Any]) -> str:
    """
    Generate content for a software/programming module with code examples.
    
    Args:
        llm: Language model instance
        module (Dict): Module data
        
    Returns:
        str: Module content in markdown format with code examples
    """
    content_prompt = PromptTemplate(
        input_variables=["title", "sections"],
        template="""
        Create detailed educational content for a programming module titled "{title}" that covers the following sections: {sections}.
        
        For each section:
        1. Provide a clear explanation of key concepts
        2. Include relevant theoretical background
        3. Explain practical applications
        4. Format the content with proper markdown headings, bullet points, and formatting
        5. Make sure the content is informative, accurate, and educational
        
        Do NOT include code examples in your response, as those will be added separately.
        
        Return the complete module content in rich markdown format:
        """
    )
    
    code_prompt = PromptTemplate(
        input_variables=["title", "sections"],
        template="""
        Create practical code examples for a programming module titled "{title}" that covers the following sections: {sections}.
        
        For each relevant concept:
        1. Provide meaningful code examples that demonstrate the concept in action
        2. Include comments that explain key parts of the code
        3. Show both basic and more advanced usage where appropriate
        4. Use proper markdown code blocks with language specification
        5. Make sure the code is correct, best practice, and would actually run
        
        Return ONLY the code examples in rich markdown format with appropriate code blocks:
        """
    )
    
    # Extract section titles for the prompt
    section_titles = []
    if 'sections' in module:
        section_titles = [section['title'] for section in module['sections']]
    
    # Create the chains
    content_chain = LLMChain(llm=llm, prompt=content_prompt)
    code_chain = LLMChain(llm=llm, prompt=code_prompt)
    
    # Run the chains
    content = content_chain.run(title=module['title'], sections=", ".join(section_titles) if section_titles else "No specific sections")
    code_examples = code_chain.run(title=module['title'], sections=", ".join(section_titles) if section_titles else "No specific sections")
    
    return SOFTWARE_MODULE_TEMPLATE.format(
        module_title=module['title'],
        module_content=content.strip(),
        code_examples=code_examples.strip()
    )


def is_software_course(category: str) -> bool:
    """
    Check if the course is a software/programming course.
    
    Args:
        category (str): Course category
        
    Returns:
        bool: True if it's a software course, False otherwise
    """
    software_keywords = [
        'software', 'programming', 'coding', 'development', 'javascript', 
        'python', 'java', 'c++', 'web', 'app', 'database', 'frontend',
        'backend', 'fullstack', 'data science', 'machine learning'
    ]
    
    return any(keyword in category.lower() for keyword in software_keywords)