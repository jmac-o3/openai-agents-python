from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel

from openai_agents import Agent, function_tool

from ..models.validation import InventoryConstraints, ValidationResult, OEMPolicyConstraints


class ModelAllocationCheck(BaseModel):
    """Model-specific allocation check result"""
    model_code: str
    allocated: int
    requested: int
    is_valid: bool
    message: str


class ConstraintCheckAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Constraint Check Agent",
            instructions="""
            You are a specialized agent responsible for validating inventory constraints 
            against dealer requirements and OEM policies. Your role is to:
            1. Validate inventory levels against min/max requirements
            2. Check compliance with OEM allocation policies
            3. Verify space and financial constraints
            4. Assess seasonal inventory adjustments
            5. Validate model mix requirements
            6. Check order bank scheduling constraints
            """,
        )

    @function_tool
    async def validate_constraints(
        self,
        inventory_data: Dict,
        constraints: InventoryConstraints,
    ) -> ValidationResult:
        """
        Validate inventory data against defined constraints

        Args:
            inventory_data: Current inventory levels and metadata
            constraints: Defined constraint parameters

        Returns:
            ValidationResult with validation status and any violations

        """
        violations = []

        # Check minimum inventory levels
        if inventory_data["total_units"] < constraints.min_units:
            violations.append(
                f"Inventory below minimum: {inventory_data['total_units']} < {constraints.min_units}",
            )

        # Check maximum inventory levels
        if inventory_data["total_units"] > constraints.max_units:
            violations.append(
                f"Inventory exceeds maximum: {inventory_data['total_units']} > {constraints.max_units}",
            )

        # Check financial constraints
        inventory_value = inventory_data["total_value"]
        if inventory_value > constraints.max_inventory_value:
            violations.append(
                f"Inventory value exceeds limit: ${inventory_value:,.2f} > ${constraints.max_inventory_value:,.2f}",
            )

        # Check space constraints
        required_space = inventory_data["total_units"] * constraints.space_per_unit
        if required_space > constraints.available_space:
            violations.append(
                f"Space requirement exceeds capacity: {required_space} > {constraints.available_space}",
            )

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)

    @function_tool
    async def validate_oem_policies(
        self,
        order_request: Dict,
        policy_constraints: OEMPolicyConstraints,
        current_date: datetime
    ) -> ValidationResult:
        """
        Validate order requests against OEM policy constraints

        Args:
            order_request: Order details including model mix and quantities
            policy_constraints: OEM policy rules and limits
            current_date: Current date for seasonal checks

        Returns:
            ValidationResult with policy compliance status
        """
        violations = []
        warnings = []

        # Check model allocation limits
        model_checks: List[ModelAllocationCheck] = []
        for model, qty in order_request["model_quantities"].items():
            allocation = policy_constraints.model_allocations.get(model, 0)
            check = ModelAllocationCheck(
                model_code=model,
                allocated=allocation,
                requested=qty,
                is_valid=qty <= allocation,
                message=f"Model {model}: Requested {qty}, Allocated {allocation}"
            )
            if not check.is_valid:
                violations.append(f"Exceeds allocation for {model}: {qty} > {allocation}")
            model_checks.append(check)

        # Check seasonal model mix requirements
        current_month = current_date.month
        if current_month in policy_constraints.seasonal_requirements:
            season_reqs = policy_constraints.seasonal_requirements[current_month]
            for model_type, required_pct in season_reqs.items():
                actual_pct = order_request["model_mix_percentages"].get(model_type, 0)
                if actual_pct < required_pct:
                    warnings.append(
                        f"Seasonal mix warning: {model_type} at {actual_pct}% (required {required_pct}%)"
                    )

        # Check minimum days supply
        if order_request["projected_days_supply"] < policy_constraints.min_days_supply:
            violations.append(
                f"Insufficient days supply: {order_request['projected_days_supply']} < {policy_constraints.min_days_supply}"
            )

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )
