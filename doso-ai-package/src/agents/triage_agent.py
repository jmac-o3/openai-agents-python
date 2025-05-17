"""
Triage Agent for DOSO AI

This agent is responsible for:
1. Analyzing incoming requests
2. Routing requests to specialized agents
3. Managing multi-agent workflows
4. Coordinating responses across agents
"""

from typing import Dict, List, Optional, Union

from openai_agents import Agent, function_tool, message

from ..models.validation import RequestValidation
from ..schemas import DealerRequest, TriageResponse, WorkflowType


class TriageAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Triage Agent",
            instructions="""
            You are an expert triage agent for the Dealer Inventory Optimization System (DOSO) AI.
            Your responsibilities include:
            
            1. Analyzing incoming dealer requests to understand their intent and needs
            2. Validating request data for completeness and correctness
            3. Determining which specialized agents should handle the request
            4. Creating appropriate workflow sequences based on request type
            5. Coordinating complex multi-agent workflows
            6. Ensuring successful completion of all request processing
            7. Providing clear, actionable guidance to dealers
            
            You must carefully analyze each request to determine the most appropriate 
            specialized agents to involve. Common request types include:
            
            - Inventory mix optimization
            - Order planning and allocation
            - Market trend analysis
            - Constraint validation
            - Sales velocity forecasting
            - Gap analysis
            
            Always prioritize accuracy, efficiency, and dealer satisfaction in your workflows.
            """,
        )

    @function_tool
    async def analyze_request(self, request: DealerRequest) -> RequestValidation:
        """
        Analyze and validate an incoming dealer request
        
        Args:
            request: The dealer request to analyze
            
        Returns:
            ValidationResult with completeness check and any missing data
        """
        # Check for required fields based on request type
        required_fields = self._get_required_fields(request.request_type)
        missing_fields = []
        
        # Check each required field
        for field in required_fields:
            if field not in request.data or not request.data[field]:
                missing_fields.append(field)
        
        return RequestValidation(
            is_complete=len(missing_fields) == 0,
            missing_fields=missing_fields,
            request_type=request.request_type,
            dealer_id=request.dealer_id,
        )

    @function_tool
    async def determine_workflow(
        self, 
        request: DealerRequest,
        validation: RequestValidation
    ) -> WorkflowType:
        """
        Determine the appropriate workflow type for the request
        
        Args:
            request: The dealer request
            validation: Validation results for the request
            
        Returns:
            WorkflowType indicating the appropriate processing workflow
        """
        # Return appropriate workflow type based on request
        if not validation.is_complete:
            return WorkflowType.DATA_COLLECTION
        
        # Map request types to workflows
        workflow_mapping = {
            "inventory_optimization": WorkflowType.INVENTORY_OPTIMIZATION,
            "order_planning": WorkflowType.ORDER_PLANNING,
            "market_analysis": WorkflowType.MARKET_ANALYSIS,
            "allocation_tracking": WorkflowType.ALLOCATION_TRACKING,
            "gap_analysis": WorkflowType.GAP_ANALYSIS,
            "constraint_check": WorkflowType.CONSTRAINT_CHECK,
        }
        
        # Default to general analysis if specific type not found
        return workflow_mapping.get(request.request_type, WorkflowType.GENERAL_ANALYSIS)

    @function_tool
    async def create_agent_sequence(
        self,
        workflow_type: WorkflowType,
        request_data: Dict
    ) -> List[str]:
        """
        Create a sequence of agent calls based on workflow type
        
        Args:
            workflow_type: The determined workflow type
            request_data: The request data
            
        Returns:
            List of agent names in sequence order
        """
        # Define agent sequences for different workflow types
        sequences = {
            WorkflowType.INVENTORY_OPTIMIZATION: [
                "inventory_analysis", 
                "market_analysis", 
                "gap_analysis"
            ],
            WorkflowType.ORDER_PLANNING: [
                "inventory_analysis", 
                "order_bank_agent", 
                "constraint_check", 
                "guidance_agent"
            ],
            WorkflowType.MARKET_ANALYSIS: [
                "market_analysis", 
                "sva_agent"
            ],
            WorkflowType.ALLOCATION_TRACKING: [
                "allocation_tracking", 
                "inventory_analysis"
            ],
            WorkflowType.GAP_ANALYSIS: [
                "gap_analysis", 
                "market_analysis", 
                "guidance_agent"
            ],
            WorkflowType.CONSTRAINT_CHECK: [
                "constraint_check", 
                "order_bank_agent"
            ],
            WorkflowType.DATA_COLLECTION: [
                "guidance_agent"
            ],
            WorkflowType.GENERAL_ANALYSIS: [
                "inventory_analysis", 
                "market_analysis", 
                "guidance_agent"
            ],
        }
        
        return sequences.get(workflow_type, ["guidance_agent"])

    @function_tool
    async def generate_triage_response(
        self,
        validation: RequestValidation,
        workflow_type: WorkflowType,
        agent_sequence: List[str],
        additional_info: Optional[str] = None,
    ) -> TriageResponse:
        """
        Generate the triage response with processing plan
        
        Args:
            validation: Validation results for the request
            workflow_type: Determined workflow type
            agent_sequence: Sequence of agents to process request
            additional_info: Any additional information or notes
            
        Returns:
            TriageResponse with request processing plan
        """
        # Generate a response with the processing plan
        return TriageResponse(
            is_valid=validation.is_complete,
            workflow_type=workflow_type,
            agent_sequence=agent_sequence,
            missing_fields=validation.missing_fields,
            additional_info=additional_info or "",
            needs_human_review=self._needs_human_review(validation, workflow_type),
        )
    
    @message
    async def process_request(self, request: DealerRequest) -> TriageResponse:
        """
        Process an incoming dealer request and determine handling approach
        
        Args:
            request: The dealer request to process
            
        Returns:
            TriageResponse with complete handling plan
        """
        # Validate the request
        validation = await self.analyze_request(request)
        
        # Determine the workflow
        workflow_type = await self.determine_workflow(request, validation)
        
        # Create agent sequence
        agent_sequence = await self.create_agent_sequence(workflow_type, request.data)
        
        # Generate additional information if needed
        additional_info = None
        if not validation.is_complete:
            missing_fields = ", ".join(validation.missing_fields)
            additional_info = f"Request requires additional data: {missing_fields}"
        
        # Generate and return response
        return await self.generate_triage_response(
            validation, 
            workflow_type, 
            agent_sequence, 
            additional_info
        )
    
    def _get_required_fields(self, request_type: str) -> List[str]:
        """Determine required fields based on request type"""
        # Define required fields for each request type
        required_fields_map = {
            "inventory_optimization": [
                "current_inventory", 
                "sales_history", 
                "dealer_id", 
                "time_period"
            ],
            "order_planning": [
                "allocation", 
                "current_inventory", 
                "constraints", 
                "sales_forecast"
            ],
            "market_analysis": [
                "market_area", 
                "competitors", 
                "time_period", 
                "segment_focus"
            ],
            "allocation_tracking": [
                "allocation_history", 
                "order_history", 
                "time_period"
            ],
            "gap_analysis": [
                "current_inventory", 
                "market_demand", 
                "competitor_offerings"
            ],
            "constraint_check": [
                "order_requirements", 
                "constraints", 
                "production_schedule"
            ],
        }
        
        return required_fields_map.get(request_type, ["dealer_id", "request_details"])
    
    def _needs_human_review(
        self, 
        validation: RequestValidation,
        workflow_type: WorkflowType
    ) -> bool:
        """Determine if request needs human review"""
        # Criteria for human review:
        # 1. Special workflow types that always require review
        special_workflows = [
            WorkflowType.CUSTOM_ANALYSIS, 
            WorkflowType.EXCEPTION_HANDLING
        ]
        
        if workflow_type in special_workflows:
            return True
            
        # 2. Incomplete requests with specific missing fields
        critical_fields = ["constraints", "dealer_id", "allocation"]
        if not validation.is_complete:
            for field in validation.missing_fields:
                if field in critical_fields:
                    return True
        
        return False
