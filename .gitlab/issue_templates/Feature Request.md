**NOTE:** If it is a feature request related to a specific customer, use `[Customer Requirement - XX, YY]` as its title prefix.

**NOTE:** If the description is empty or the user story is ill-defined, the issue will be marked as `status::Needs Definition`. The title should be clear enough as well.

**NOTE:** Label the issue with `feature::*` and/or `component::*` at least.

**NOTE:** If there are any GitLab issues which prevent this issue from being started/completed, link them as `is blocked by`.


**User Story**
- A clear but concise requirement definition from **an user perspetive**. An user can be a customer, a team member, MLPerf team, another Merlin component, etc.  Make the story independent and small if possible while specifying the **problem** and **goal** clearly.
  - Example 1. As our customer XX and YY put their data in HDFS, I will extend the HugeCTR DataReader to load the HDFS-resident dataset. 
  - Example 2. Because the the MLPerf vx.y submission uses the model XX which includes the layer YY, we'd like to implement the GPU acccelerated YY layer.

**Use Cases**
- (If applicable) A pesudo code (Python or C++) level description of how it is used or intercts with other components.

**Test Cases**
- (If applicable) Describe the utest and integration test while providng the test data, input parameters, test steps and expected results.


**Design Document**
- (If applicable)

**Task List**
- Provide the atomic tasks required to complete the issue.
