"""Code generation tools with LLM and ML integration.

Provides tools for generating, analyzing, and optimizing code using
LLMs and machine learning models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from loguru import logger
import os
import json
import ast
import subprocess
from pathlib import Path

try:
    from swarms.structs.agent import Agent
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False
    logger.warning("swarms not available - LLM features will be limited")


class BaseCodeGenerator(ABC):
    """Abstract base class for code generators."""
    
    @abstractmethod
    def generate(self, specification: str, language: str = "python", **kwargs) -> str:
        """
        Generate code from specification.
        
        Args:
            specification: Natural language or structured specification
            language: Target programming language
            **kwargs: Additional parameters
            
        Returns:
            str: Generated code
        """
        pass
    
    @abstractmethod
    def validate(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Validate generated code.
        
        Args:
            code: Code to validate
            language: Programming language
            
        Returns:
            Dict with validation results
        """
        pass


class LLMCodeGenerator(BaseCodeGenerator):
    """LLM-powered code generator using swarms Agent."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize LLM code generator.
        
        Args:
            model_name: LLM model to use
            system_prompt: Custom system prompt
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.verbose = verbose
        
        if not SWARMS_AVAILABLE:
            raise ImportError("swarms is required for LLMCodeGenerator")
        
        default_prompt = """You are an expert code generator. Generate clean, functional, 
well-documented code based on specifications. Always include error handling and follow 
best practices for the target language."""
        
        self.agent = Agent(
            agent_name="CodeGenerator",
            system_prompt=system_prompt or default_prompt,
            model_name=model_name,
            max_loops=3,
            verbose=verbose
        )
    
    def generate(
        self,
        specification: str,
        language: str = "python",
        include_tests: bool = True,
        include_docs: bool = True,
        **kwargs
    ) -> str:
        """Generate code using LLM."""
        prompt = f"""Generate {language} code for the following specification:

{specification}

Requirements:
- Language: {language}
- Include tests: {include_tests}
- Include documentation: {include_docs}
- Follow best practices
- Include error handling
- Make it production-ready

Generate only the code, no explanations unless requested."""
        
        try:
            result = self.agent.run(prompt)
            if isinstance(result, dict):
                code = result.get('code', result.get('response', str(result)))
            else:
                code = str(result)
            
            # Extract code from markdown code blocks if present
            if "```" in code:
                lines = code.split("\n")
                code_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        code_lines.append(line)
                code = "\n".join(code_lines)
            
            return code.strip()
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    def validate(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code using syntax checking."""
        validation = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "language": language
        }
        
        if language == "python":
            try:
                ast.parse(code)
                validation["valid"] = True
            except SyntaxError as e:
                validation["errors"].append({
                    "type": "syntax_error",
                    "message": str(e),
                    "line": e.lineno,
                    "offset": e.offset
                })
            except Exception as e:
                validation["errors"].append({
                    "type": "parse_error",
                    "message": str(e)
                })
        
        return validation
    
    def optimize(self, code: str, language: str = "python") -> str:
        """
        Optimize code using LLM suggestions.
        
        Args:
            code: Code to optimize
            language: Programming language
            
        Returns:
            str: Optimized code
        """
        prompt = f"""Optimize the following {language} code for:
- Performance
- Readability
- Best practices
- Error handling

Original code:
```{language}
{code}
```

Return only the optimized code, no explanations."""
        
        try:
            result = self.agent.run(prompt)
            if isinstance(result, dict):
                optimized = result.get('code', result.get('response', str(result)))
            else:
                optimized = str(result)
            
            # Extract code from markdown
            if "```" in optimized:
                lines = optimized.split("\n")
                code_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        code_lines.append(line)
                optimized = "\n".join(code_lines)
            
            return optimized.strip()
        except Exception as e:
            logger.error(f"Code optimization failed: {e}")
            return code  # Return original on failure


class MLCodeAnalyzer(BaseCodeGenerator):
    """ML-powered code analyzer for patterns and quality."""
    
    def __init__(self, use_llm: bool = True, model_name: str = "gpt-4o-mini"):
        """
        Initialize ML code analyzer.
        
        Args:
            use_llm: Whether to use LLM for analysis
            model_name: LLM model name if use_llm is True
        """
        self.use_llm = use_llm
        if use_llm and SWARMS_AVAILABLE:
            self.agent = Agent(
                agent_name="CodeAnalyzer",
                system_prompt="You are an expert code analyzer. Analyze code for quality, patterns, security issues, and best practices.",
                model_name=model_name,
                max_loops=2,
                verbose=False
            )
        else:
            self.agent = None
    
    def generate(self, specification: str, language: str = "python", **kwargs) -> str:
        """Not implemented - this is an analyzer, not a generator."""
        raise NotImplementedError("MLCodeAnalyzer is for analysis, not generation")
    
    def validate(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code with ML analysis."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "quality_score": 0.0,
            "security_issues": [],
            "performance_issues": [],
            "best_practices": []
        }
        
        # Basic syntax validation
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                validation["valid"] = False
                validation["errors"].append({
                    "type": "syntax_error",
                    "message": str(e),
                    "line": e.lineno
                })
                return validation
        
        # LLM-based quality analysis
        if self.use_llm and self.agent:
            try:
                analysis_prompt = f"""Analyze this {language} code for:
1. Code quality and maintainability
2. Security vulnerabilities
3. Performance issues
4. Best practice violations
5. Overall quality score (0-100)

Code:
```{language}
{code}
```

Return JSON with: quality_score, security_issues (list), performance_issues (list), best_practices (list), warnings (list)."""
                
                result = self.agent.run(analysis_prompt)
                if isinstance(result, dict):
                    analysis = result
                else:
                    # Try to parse JSON from string
                    try:
                        analysis = json.loads(str(result))
                    except:
                        analysis = {"quality_score": 70.0}
                
                validation["quality_score"] = analysis.get("quality_score", 70.0)
                validation["security_issues"] = analysis.get("security_issues", [])
                validation["performance_issues"] = analysis.get("performance_issues", [])
                validation["best_practices"] = analysis.get("best_practices", [])
                validation["warnings"].extend(analysis.get("warnings", []))
                
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        return validation
    
    def analyze_patterns(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Analyze code for design patterns and structure.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Dict with pattern analysis
        """
        patterns = {
            "design_patterns": [],
            "complexity": "medium",
            "structure": "unknown",
            "dependencies": []
        }
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                # Detect patterns
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        patterns["design_patterns"].append("class_definition")
                    if isinstance(node, ast.FunctionDef):
                        if any(isinstance(n, ast.Yield) for n in ast.walk(node)):
                            patterns["design_patterns"].append("generator")
                
                # Count complexity
                function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                
                if function_count > 20 or class_count > 10:
                    patterns["complexity"] = "high"
                elif function_count < 5 and class_count < 3:
                    patterns["complexity"] = "low"
                
            except Exception as e:
                logger.warning(f"Pattern analysis failed: {e}")
        
        return patterns


class HybridCodeGenerator(BaseCodeGenerator):
    """Hybrid code generator combining LLM and ML techniques."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        use_ml_optimization: bool = True,
        verbose: bool = False
    ):
        """
        Initialize hybrid code generator.
        
        Args:
            model_name: LLM model name
            use_ml_optimization: Enable ML-based optimization
            verbose: Enable verbose logging
        """
        self.generator = LLMCodeGenerator(model_name=model_name, verbose=verbose)
        self.analyzer = MLCodeAnalyzer(use_llm=True, model_name=model_name) if use_ml_optimization else None
        self.verbose = verbose
    
    def generate(
        self,
        specification: str,
        language: str = "python",
        iterations: int = 2,
        **kwargs
    ) -> str:
        """
        Generate code with iterative improvement.
        
        Args:
            specification: Code specification
            language: Target language
            iterations: Number of improvement iterations
            **kwargs: Additional parameters
            
        Returns:
            str: Generated and optimized code
        """
        # Initial generation
        code = self.generator.generate(specification, language, **kwargs)
        
        if self.analyzer and iterations > 1:
            # Iterative improvement
            for i in range(iterations - 1):
                validation = self.analyzer.validate(code, language)
                
                if validation["quality_score"] >= 90:
                    if self.verbose:
                        logger.info(f"Code quality sufficient (score: {validation['quality_score']})")
                    break
                
                # Optimize based on analysis
                if validation["warnings"] or validation["security_issues"]:
                    optimization_spec = f"""Original code has these issues:
- Warnings: {validation['warnings']}
- Security issues: {validation['security_issues']}
- Quality score: {validation['quality_score']}

Please improve the code to address these issues."""
                    
                    code = self.generator.generate(
                        f"{specification}\n\n{optimization_spec}",
                        language,
                        **kwargs
                    )
        
        return code
    
    def validate(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate using both generators."""
        validation = self.generator.validate(code, language)
        
        if self.analyzer:
            ml_validation = self.analyzer.validate(code, language)
            validation.update(ml_validation)
        
        return validation


class CodeGeneratorRegistry:
    """Registry for code generators."""
    
    _generators: Dict[str, BaseCodeGenerator] = {}
    
    @classmethod
    def register(cls, name: str, generator: BaseCodeGenerator):
        """Register a code generator."""
        cls._generators[name] = generator
        logger.info(f"Registered code generator: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseCodeGenerator]:
        """Get a registered generator."""
        return cls._generators.get(name)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered generators."""
        return list(cls._generators.keys())
    
    @classmethod
    def create_default(cls) -> HybridCodeGenerator:
        """Create and register default generator."""
        default = HybridCodeGenerator()
        cls.register("default", default)
        return default

