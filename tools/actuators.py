"""Actuator interfaces and implementations for policy engine.

Provides standardized actuator interfaces for executing interventions
on system controls, cloud resources, and services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import logger
import subprocess
import os

from schemas.policy import InterventionSpec


class BaseActuator(ABC):
    """Abstract base class for actuators.
    
    All actuators must implement execute() to apply interventions.
    """
    
    @abstractmethod
    def execute(self, interventions: List[InterventionSpec]) -> Dict[str, Any]:
        """
        Execute interventions.
        
        Args:
            interventions: List of intervention specifications
            
        Returns:
            Dict[str, Any]: Execution results with status and metadata
        """
        pass
    
    def validate(self, intervention: InterventionSpec) -> bool:
        """
        Validate intervention before execution.
        
        Args:
            intervention: Intervention to validate
            
        Returns:
            bool: True if intervention is valid
        """
        return True
    
    def rollback(self, rollback_descriptor: Dict[str, Any]) -> bool:
        """
        Rollback an intervention.
        
        Args:
            rollback_descriptor: Rollback descriptor from intervention
            
        Returns:
            bool: True if rollback succeeded
        """
        logger.warning(f"Rollback not implemented for {self.__class__.__name__}")
        return False
    
    def status(self) -> Dict[str, Any]:
        """
        Get actuator status.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {"status": "unknown"}


class SystemControlActuator(BaseActuator):
    """OS-level system control actuator.
    
    Supports:
    - CPU throttling (nice values, cgroups)
    - Process management (start/stop/kill)
    - I/O prioritization (ionice)
    - Network QoS (tc, iptables) - requires root
    """
    
    def __init__(self, require_root: bool = False):
        """
        Initialize system control actuator.
        
        Args:
            require_root: Whether to require root privileges
        """
        self.require_root = require_root
        if require_root and os.geteuid() != 0:
            logger.warning("Root privileges may be required for some operations")
    
    def execute(self, interventions: List[InterventionSpec]) -> Dict[str, Any]:
        """Execute system control interventions."""
        results = []
        
        for intervention in interventions:
            lever_id = intervention.lever_id
            params = intervention.parameters
            
            try:
                if lever_id == "cpu_throttle":
                    result = self._cpu_throttle(params)
                elif lever_id == "process_control":
                    result = self._process_control(params)
                elif lever_id == "io_priority":
                    result = self._io_priority(params)
                else:
                    result = {"status": "error", "message": f"Unknown lever: {lever_id}"}
                
                results.append({
                    "intervention": intervention.model_dump(),
                    "result": result
                })
            except Exception as e:
                logger.error(f"Actuator execution failed for {lever_id}: {e}")
                results.append({
                    "intervention": intervention.model_dump(),
                    "result": {"status": "error", "message": str(e)}
                })
        
        return {
            "executed": len(results),
            "results": results
        }
    
    def _cpu_throttle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CPU throttling via nice values."""
        nice_value = params.get("nice_value", 0)
        pid = params.get("pid")
        
        if pid is None:
            return {"status": "error", "message": "pid required for cpu_throttle"}
        
        try:
            # Set nice value (higher = lower priority)
            os.nice(nice_value - os.nice(0))  # Adjust from current nice value
            return {"status": "success", "nice_value": nice_value, "pid": pid}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _process_control(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Control process (start/stop/kill)."""
        action = params.get("action")  # "start", "stop", "kill"
        pid = params.get("pid")
        command = params.get("command")
        
        try:
            if action == "start" and command:
                # Start process
                process = subprocess.Popen(
                    command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                return {"status": "success", "action": action, "pid": process.pid}
            
            elif action in ["stop", "kill"] and pid:
                # Stop or kill process
                os.kill(pid, 15 if action == "stop" else 9)
                return {"status": "success", "action": action, "pid": pid}
            
            else:
                return {"status": "error", "message": "Invalid parameters for process_control"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _io_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set I/O priority using ionice (requires ionice command)."""
        priority_class = params.get("priority_class", 2)  # 1=realtime, 2=best-effort, 3=idle
        priority_level = params.get("priority_level", 4)  # 0-7 for best-effort
        pid = params.get("pid")
        
        if pid is None:
            return {"status": "error", "message": "pid required for io_priority"}
        
        try:
            # Use ionice command if available
            cmd = ["ionice", "-c", str(priority_class), "-n", str(priority_level), "-p", str(pid)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return {"status": "success", "priority_class": priority_class, "priority_level": priority_level, "pid": pid}
            else:
                return {"status": "error", "message": result.stderr}
        except FileNotFoundError:
            return {"status": "error", "message": "ionice command not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def rollback(self, rollback_descriptor: Dict[str, Any]) -> bool:
        """Rollback system control intervention."""
        lever_id = rollback_descriptor.get("lever_id")
        params = rollback_descriptor.get("parameters", {})
        
        try:
            if lever_id == "cpu_throttle":
                # Reset to default nice value (0)
                params["nice_value"] = 0
                result = self._cpu_throttle(params)
                return result.get("status") == "success"
            elif lever_id == "process_control":
                # For process control, rollback might mean restarting stopped process
                # This is context-dependent, so we just log
                logger.info(f"Rollback for process_control requires manual intervention")
                return True
            else:
                logger.warning(f"Rollback not implemented for {lever_id}")
                return False
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


class CloudResourceActuator(BaseActuator):
    """Abstract base for cloud provider resource controls.
    
    Subclasses should implement provider-specific control operations.
    """
    
    def __init__(self, provider: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cloud resource actuator.
        
        Args:
            provider: Cloud provider name ("aws", "gcp", "azure")
            config: Provider-specific configuration
        """
        self.provider = provider
        self.config = config or {}
    
    @abstractmethod
    def execute(self, interventions: List[InterventionSpec]) -> Dict[str, Any]:
        """Execute cloud resource controls (implemented by subclasses)."""
        pass


class ServiceActuator(BaseActuator):
    """Service management actuator (systemd, Docker, etc.)."""
    
    def __init__(self, service_type: str = "systemd"):
        """
        Initialize service actuator.
        
        Args:
            service_type: Type of service manager ("systemd", "docker", "kubernetes")
        """
        self.service_type = service_type
    
    def execute(self, interventions: List[InterventionSpec]) -> Dict[str, Any]:
        """Execute service control interventions."""
        results = []
        
        for intervention in interventions:
            lever_id = intervention.lever_id
            params = intervention.parameters
            
            try:
                if lever_id == "service_control":
                    result = self._service_control(params)
                else:
                    result = {"status": "error", "message": f"Unknown lever: {lever_id}"}
                
                results.append({
                    "intervention": intervention.model_dump(),
                    "result": result
                })
            except Exception as e:
                logger.error(f"Service actuator execution failed: {e}")
                results.append({
                    "intervention": intervention.model_dump(),
                    "result": {"status": "error", "message": str(e)}
                })
        
        return {
            "executed": len(results),
            "results": results
        }
    
    def _service_control(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Control systemd service."""
        action = params.get("action")  # "start", "stop", "restart", "reload"
        service_name = params.get("service_name")
        
        if not service_name:
            return {"status": "error", "message": "service_name required"}
        
        try:
            if self.service_type == "systemd":
                cmd = ["systemctl", action, service_name]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    return {"status": "success", "action": action, "service": service_name}
                else:
                    return {"status": "error", "message": result.stderr}
            else:
                return {"status": "error", "message": f"Service type {self.service_type} not implemented"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class ActuatorRegistry:
    """Registry mapping lever types to actuator instances.
    
    Provides bounds validation and transaction support.
    """
    
    def __init__(self):
        """Initialize actuator registry."""
        self.actuators: Dict[str, BaseActuator] = {}
        self.lever_to_actuator: Dict[str, str] = {}  # lever_id -> actuator_key
    
    def register(self, actuator_key: str, actuator: BaseActuator, lever_types: Optional[List[str]] = None) -> None:
        """
        Register an actuator.
        
        Args:
            actuator_key: Unique identifier for actuator
            actuator: Actuator instance
            lever_types: Optional list of lever IDs this actuator handles
        """
        self.actuators[actuator_key] = actuator
        if lever_types:
            for lever_id in lever_types:
                self.lever_to_actuator[lever_id] = actuator_key
        logger.info(f"Registered actuator: {actuator_key}")
    
    def get_actuator(self, actuator_key: str) -> Optional[BaseActuator]:
        """Get actuator by key."""
        return self.actuators.get(actuator_key)
    
    def get_actuator_for_lever(self, lever_id: str) -> Optional[BaseActuator]:
        """Get actuator for a specific lever."""
        actuator_key = self.lever_to_actuator.get(lever_id)
        if actuator_key:
            return self.actuators.get(actuator_key)
        return None
    
    def execute(self, interventions: List[InterventionSpec], transaction: bool = False) -> Dict[str, Any]:
        """
        Execute interventions via appropriate actuators.
        
        Args:
            interventions: List of interventions to execute
            transaction: If True, execute all-or-nothing (rollback on failure)
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if transaction:
            # Transaction mode: collect all results, rollback on any failure
            results = []
            executed = []
            
            for intervention in interventions:
                actuator = self.get_actuator_for_lever(intervention.lever_id)
                if not actuator:
                    # Rollback all previous
                    for prev_intervention in executed:
                        self._rollback_intervention(prev_intervention)
                    return {
                        "status": "error",
                        "message": f"No actuator for lever {intervention.lever_id}",
                        "results": results
                    }
                
                # Validate before execution
                if not actuator.validate(intervention):
                    # Rollback all previous
                    for prev_intervention in executed:
                        self._rollback_intervention(prev_intervention)
                    return {
                        "status": "error",
                        "message": f"Validation failed for {intervention.lever_id}",
                        "results": results
                    }
                
                # Execute
                result = actuator.execute([intervention])
                results.append(result)
                executed.append(intervention)
                
                # Check for errors
                if result.get("results", [{}])[0].get("result", {}).get("status") == "error":
                    # Rollback all previous
                    for prev_intervention in executed:
                        self._rollback_intervention(prev_intervention)
                    return {
                        "status": "error",
                        "message": f"Execution failed for {intervention.lever_id}",
                        "results": results
                    }
            
            return {
                "status": "success",
                "executed": len(executed),
                "results": results
            }
        else:
            # Non-transaction mode: execute independently
            all_results = []
            for intervention in interventions:
                actuator = self.get_actuator_for_lever(intervention.lever_id)
                if actuator:
                    if actuator.validate(intervention):
                        result = actuator.execute([intervention])
                        all_results.append(result)
                    else:
                        all_results.append({
                            "status": "error",
                            "message": f"Validation failed for {intervention.lever_id}"
                        })
                else:
                    all_results.append({
                        "status": "error",
                        "message": f"No actuator for lever {intervention.lever_id}"
                    })
            
            return {
                "status": "partial",
                "executed": len(all_results),
                "results": all_results
            }
    
    def _rollback_intervention(self, intervention: InterventionSpec) -> bool:
        """Rollback a single intervention."""
        if not intervention.rollback_descriptor:
            return False
        
        actuator = self.get_actuator_for_lever(intervention.lever_id)
        if actuator:
            return actuator.rollback(intervention.rollback_descriptor)
        return False
    
    def status_all(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all actuators."""
        return {key: actuator.status() for key, actuator in self.actuators.items()}

