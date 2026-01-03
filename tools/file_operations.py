"""File operations tools with LLM and ML integration.

Provides tools for file manipulation, content generation, and intelligent
file operations using LLMs and ML models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from loguru import logger
import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime

try:
    from swarms.structs.agent import Agent
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False
    logger.warning("swarms not available - LLM features will be limited")


class BaseFileOperator(ABC):
    """Abstract base class for file operations."""
    
    @abstractmethod
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute file operation.
        
        Args:
            operation: Operation type
            **kwargs: Operation parameters
            
        Returns:
            Dict with operation results
        """
        pass


class FileWriter(BaseFileOperator):
    """Basic file writing operations."""
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute file operation."""
        if operation == "write":
            return self.write_file(**kwargs)
        elif operation == "append":
            return self.append_file(**kwargs)
        elif operation == "create_directory":
            return self.create_directory(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    
    def write_file(
        self,
        filepath: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write content to file.
        
        Args:
            filepath: Path to file
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if needed
            
        Returns:
            Dict with write results
        """
        try:
            path = Path(filepath)
            
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return {
                "success": True,
                "filepath": str(path),
                "size": len(content.encode(encoding)),
                "created": path.exists()
            }
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return {"success": False, "error": str(e)}
    
    def append_file(
        self,
        filepath: str,
        content: str,
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """Append content to file."""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'a', encoding=encoding) as f:
                f.write(content)
            
            return {
                "success": True,
                "filepath": str(path),
                "appended_size": len(content.encode(encoding))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_directory(self, dirpath: str) -> Dict[str, Any]:
        """Create directory."""
        try:
            path = Path(dirpath)
            path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "dirpath": str(path), "created": path.exists()}
        except Exception as e:
            return {"success": False, "error": str(e)}


class LLMFileGenerator(BaseFileOperator):
    """LLM-powered file content generation."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        base_writer: Optional[FileWriter] = None,
        verbose: bool = False
    ):
        """
        Initialize LLM file generator.
        
        Args:
            model_name: LLM model name
            base_writer: Base file writer instance
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.writer = base_writer or FileWriter()
        self.verbose = verbose
        
        if SWARMS_AVAILABLE:
            self.agent = Agent(
                agent_name="FileGenerator",
                system_prompt="You are an expert at generating file content. Generate well-structured, complete content based on specifications.",
                model_name=model_name,
                max_loops=2,
                verbose=verbose
            )
        else:
            self.agent = None
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute file operation with LLM generation."""
        if operation == "generate_and_write":
            return self.generate_and_write(**kwargs)
        elif operation == "generate_content":
            return {"success": True, "content": self.generate_content(**kwargs)}
        else:
            return self.writer.execute(operation, **kwargs)
    
    def generate_content(
        self,
        specification: str,
        file_type: str = "text",
        format_hints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate file content using LLM.
        
        Args:
            specification: Content specification
            file_type: Type of file (text, json, yaml, code, markdown, etc.)
            format_hints: Formatting hints
            
        Returns:
            str: Generated content
        """
        if not self.agent:
            raise RuntimeError("LLM agent not available")
        
        format_instructions = {
            "json": "Generate valid JSON format",
            "yaml": "Generate valid YAML format",
            "code": "Generate code with proper syntax",
            "markdown": "Generate Markdown format",
            "text": "Generate plain text"
        }
        
        prompt = f"""Generate {file_type} file content based on this specification:

{specification}

Requirements:
- Format: {format_instructions.get(file_type, 'text')}
- Complete and functional
- Well-structured
- Production-ready

{f'Format hints: {format_hints}' if format_hints else ''}

Generate only the content, no explanations."""
        
        try:
            result = self.agent.run(prompt)
            if isinstance(result, dict):
                content = result.get('content', result.get('response', str(result)))
            else:
                content = str(result)
            
            # Clean up markdown code blocks if present
            if "```" in content:
                lines = content.split("\n")
                content_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        content_lines.append(line)
                content = "\n".join(content_lines)
            
            return content.strip()
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise
    
    def generate_and_write(
        self,
        filepath: str,
        specification: str,
        file_type: str = "text",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content and write to file.
        
        Args:
            filepath: Target file path
            specification: Content specification
            file_type: Type of file
            **kwargs: Additional parameters
            
        Returns:
            Dict with operation results
        """
        try:
            content = self.generate_content(specification, file_type, **kwargs)
            result = self.writer.write_file(filepath, content, **kwargs)
            result["generated"] = True
            result["content_length"] = len(content)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


class IntelligentFileManager(BaseFileOperator):
    """Intelligent file manager with ML-based organization."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        use_llm: bool = True
    ):
        """
        Initialize intelligent file manager.
        
        Args:
            model_name: LLM model name
            use_llm: Enable LLM features
        """
        self.writer = FileWriter()
        self.generator = LLMFileGenerator(model_name=model_name) if use_llm and SWARMS_AVAILABLE else None
        self.use_llm = use_llm and SWARMS_AVAILABLE
        
        if self.use_llm:
            self.agent = Agent(
                agent_name="FileManager",
                system_prompt="You are an expert file organizer. Analyze file structures and suggest optimal organization.",
                model_name=model_name,
                max_loops=2,
                verbose=False
            )
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute intelligent file operation."""
        if operation == "organize_directory":
            return self.organize_directory(**kwargs)
        elif operation == "suggest_structure":
            return {"success": True, "suggestion": self.suggest_structure(**kwargs)}
        elif operation == "generate_project_structure":
            return self.generate_project_structure(**kwargs)
        else:
            return self.writer.execute(operation, **kwargs)
    
    def suggest_structure(
        self,
        project_description: str,
        project_type: str = "software"
    ) -> Dict[str, Any]:
        """
        Suggest project file structure using LLM.
        
        Args:
            project_description: Project description
            project_type: Type of project
            
        Returns:
            Dict with suggested structure
        """
        if not self.use_llm:
            return {
                "structure": ["src/", "tests/", "docs/", "README.md"],
                "reasoning": "Default structure"
            }
        
        prompt = f"""Suggest an optimal file structure for a {project_type} project:

{project_description}

Provide:
1. Directory structure
2. Key files needed
3. Organization rationale

Return as structured JSON with 'directories' (list), 'files' (list), and 'reasoning' (string)."""
        
        try:
            result = self.agent.run(prompt)
            if isinstance(result, dict):
                return result
            else:
                # Try to parse JSON
                try:
                    return json.loads(str(result))
                except:
                    return {
                        "structure": ["src/", "tests/", "docs/"],
                        "reasoning": "LLM response parsing failed"
                    }
        except Exception as e:
            logger.warning(f"Structure suggestion failed: {e}")
            return {
                "structure": ["src/", "tests/", "docs/"],
                "reasoning": f"Error: {str(e)}"
            }
    
    def generate_project_structure(
        self,
        base_path: str,
        project_description: str,
        project_type: str = "software"
    ) -> Dict[str, Any]:
        """
        Generate complete project structure.
        
        Args:
            base_path: Base directory path
            project_description: Project description
            project_type: Type of project
            
        Returns:
            Dict with generation results
        """
        suggestion = self.suggest_structure(project_description, project_type)
        
        created = []
        errors = []
        
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        for dir_name in suggestion.get("directories", []):
            dir_path = base / dir_name
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(str(dir_path))
            except Exception as e:
                errors.append(f"Failed to create {dir_name}: {e}")
        
        # Create files
        for file_name in suggestion.get("files", []):
            file_path = base / file_name
            try:
                if self.generator:
                    # Generate content for file
                    spec = f"Create {file_name} for {project_description}"
                    content = self.generator.generate_content(spec, file_type="text")
                    self.writer.write_file(str(file_path), content)
                else:
                    # Create empty file
                    file_path.touch()
                created.append(str(file_path))
            except Exception as e:
                errors.append(f"Failed to create {file_name}: {e}")
        
        return {
            "success": len(errors) == 0,
            "base_path": str(base),
            "created": created,
            "errors": errors,
            "structure": suggestion
        }
    
    def organize_directory(
        self,
        directory: str,
        strategy: str = "by_type"
    ) -> Dict[str, Any]:
        """
        Organize directory using intelligent strategy.
        
        Args:
            directory: Directory to organize
            strategy: Organization strategy
            
        Returns:
            Dict with organization results
        """
        # Implementation would analyze files and organize them
        # This is a simplified version
        return {
            "success": True,
            "directory": directory,
            "strategy": strategy,
            "organized": []
        }


class FileOperationsRegistry:
    """Registry for file operations."""
    
    _operators: Dict[str, BaseFileOperator] = {}
    
    @classmethod
    def register(cls, name: str, operator: BaseFileOperator):
        """Register a file operator."""
        cls._operators[name] = operator
        logger.info(f"Registered file operator: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseFileOperator]:
        """Get a registered operator."""
        return cls._operators.get(name)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered operators."""
        return list(cls._operators.keys())
    
    @classmethod
    def create_default(cls) -> IntelligentFileManager:
        """Create and register default operator."""
        default = IntelligentFileManager()
        cls.register("default", default)
        return default

