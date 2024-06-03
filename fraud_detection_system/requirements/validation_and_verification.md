# Task 4: Validation and Verification

## Task 4.1: Describe validation methods

In general, the requirements can be validated by reviewing or walking through the requirements document with all stakeholders involved including engineers, customer service representatives, CEO, legal and compliance teams, and Chief Risk Officer. However, getting all of these stakeholers into a meeting at once would be difficult, so it may be easier to have multiple short meetings with each stakeholder, focusing on the requirements that affect them directly, and asking if they think any requirements are missing. Then I, as the project lead, would compile the results of the conversations, double-checking for feasibility and completeness, iterating on this process as necessary. Validation methods for each requirement are listed below.

1. COO & Legal team want the system to follow all data handling and privacy laws.
   - The accuracy and completeness of this requirement can be validated by a walk through or review with the legal and compliance teams.
2. Engineering and legal teams want the system and design decisions to be well-documented.
   - This requirement can be validated via walkthrough with leads from the engineering and legal teams to ensure all topics to be documented are covered.
3. CEO and Chief Risk Officer want to prioritize fraud detection rate and maximize true positives.
   - This requirement can be confirmed as valid by conducting a simple requirements walk through with the CEO and/or risk officer to ensure accuracy and completeness.
4. Chief Risk Officer wants to minimize the highest risk fraud type, which is account take over.
   - This requirement can be validated by a walk through with the risk officer or their team to ensure accuracy.
5. Chief Risk Officer wants the system to prioritize reducing the amount of financial loss instead of total number of fraudulent transactions.
   - This requirement can be validated by a walk through with the risk officer or their team to ensure accuracy.
6. CEO wants new system to exceed previous metrics of 40% precision and 85% recall.
   - This requirement can be validated for accuracy and completeness by walking through the requirement with the CEO.
7. Credit card users want to receive as few transaction confirmation notifications as possible.
   - This can be validated by a walkthrough with leaders of the customer service team, and further confirmed by surveys or interviews with customers.
8. CEO wants fraud to be detected as quickly as possible when the transaction occurs.
   - This requirement can be validated for accuracy and completeness by walking through the requirement with the CEO.
9. CEO wants the system to have a flexible and customizable alert system to notify stakeholders such as fraud analysts and customer service representatives about flagged transactions.
   - This requirement can be validated for accuracy and completeness by walking through the requirement with the CEO, fraud analyst lead, and customer service representative lead.
10. CEO and Head of Credit Card Operations want the system to integrate with customer relationship management platforms, payment systems, analytic tools.
    - This requirement can be validated for accuracy and completeness by walk-through or review and inspection with the CEO and Head of Credit Card Operations.

## Task 4.2: Describe verification techniques

In general, these requirements can be verified using techniques including review and inspection by stakeholders, thorough model testing, performance metrics calculation, and post-deployment monitoring and testing to ensure continued requirement verification. Specific verification techniques for each requirement are listed below.

1. COO & Legal team want the system to follow all data handling and privacy laws.
   - Similar to validation, verification of this requirement can be achieved through review with a member from the data engineering, legal, and compliance teams to ensure all laws are followed.
2. Engineering and legal teams want the system and design decisions to be well-documented.
   - Similar to validation, verification of this requirement can be achieved through review and inspection of the documentation by members of the engineering and legal or compliance teams to ensure all details are properly documented.
3. CEO and Chief Risk Officer want to prioritize fraud detection rate and maximize true positives.
   - This requirement can be verified by testing the model with representative test data to ensure that true positive fraud detection rates are optimized.
4. Chief Risk Officer wants to minimize the highest risk fraud type, which is account take over.
   - This requirement can be verified by testing the model with data that is consistent with account takeover fraud, and ensuring model performance performs well especially on this type of data. Monitoring the system post-deployment will help ensure this requirement continues to be met.
5. Chief Risk Officer wants the system to prioritize reducing the amount of financial loss instead of total number of fraudulent transactions.
   - Similar to the above requirement, this one can be verified by testing the model with data that includes fraudulent transactions of varying monetary sizes, and ensuring optimized performance especially on the high cost incidents. Monitoring the system post-deployment will help ensure this requirement continues to be met.
6. CEO wants new system to exceed previous metrics of 40% precision and 85% recall.
   - This requirement can be validated by calculating the performance and recall metrics during model testing and fine-tuning and ensuring they meet these thresholds. Monitoring these metrics post-deployment will help ensure this requirement continues to be met.
7. Credit card users want to receive as few transaction confirmation notifications as possible.
   - This can be verified by pre-deployment user assurance testing, and post-deloyment customer feedback and customer satisfaction surveys.
8. CEO wants fraud to be detected as quickly as possible when the transaction occurs.
   - This requirement can be validated by calculating time to detection during testing, and ensuring the average time is small enough, perhaps below some emperically determined numeric threshold. Monitoring the system post-deployment will help ensure this requirement continues to be met.
9. CEO wants the system to have a flexible and customizable alert system to notify stakeholders such as fraud analysts and customer service representatives about flagged transactions.
   - This requirement can be verified by testing the alert systems before and after system deployment, ensuring that all necessary stakeholders receive alerts with adequate information, and do not receive alerts when they're not necessary (e.g. non-fraudlent transactions). Periodic alert testing post-deployment may help ensure that alerts work as expected.
10. CEO and Head of Credit Card Operations want the system to integrate with customer relationship management platforms, payment systems, analytic tools.
    - This requirement can be verified by testing all integrations before and after system deployment, ensuring that data is passed appropriately between systems. Periodic post-deployment testing may also help to ensure that integrations behave as expected.
