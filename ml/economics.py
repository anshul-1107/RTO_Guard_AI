"""
Business assumptions used to pick the decision threshold and report savings.
All values in INR. Tweak these to match a real brand.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Economics:
    rto_cost: float = 180.0            # forward + reverse shipping + packaging lost per RTO
    intervention_cost: float = 10.0    # WhatsApp + agent call-back per flagged order
    intervention_success: float = 0.40  # share of would-be RTOs prevented by intervention
    friction_loss_rate: float = 0.05   # good customers who cancel because we nudged them
    order_margin: float = 200.0        # margin lost when a good customer cancels

    def net_value(self, is_rto, flagged):
        """Per-order net INR from flagging, vs. doing nothing."""
        saved = is_rto * self.intervention_success * self.rto_cost
        friction = (1 - is_rto) * self.friction_loss_rate * self.order_margin
        return flagged * (saved - friction - self.intervention_cost)


ECON = Economics()
