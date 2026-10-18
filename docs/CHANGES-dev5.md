# Engine 0.1.0.dev5

Passive probe snapshots now include signed bilinear `field_values` and their `field_time`, including after the finite intensity window completes. Optional field arguments on configure/reset initialize samples from the current field; simulator reset supplies its initial field. Existing intensity accumulation and default CSV semantics are unchanged. The numerical solver stencil is unchanged.

Validation: 15 tests across core, measurement and grid-budget tests.
