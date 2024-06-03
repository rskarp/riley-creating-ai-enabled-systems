# Task 3: Requirements Specification

## Task 3.1: Explain requirements

### 1. COO & Legal team want the system to follow all data handling and privacy laws.

All laws regarding data privacy, handling, storage, use, access controls, annonymization, and encryption must be followed in order to prevent legal issues and potential business failure. Such laws include the General Data Protection Regulation (GDPR) or the California Consumer Privacy Act, Fair Credit Reporting Act (FCRA), Equal Credit Opportunity Act (ECOA), and various anti-discrimination laws. This requirement can be addressed by collaborating with the COO, legal team, compliance officers, and data engineering team to ensure that all data laws are being followed.

### 2. Engineering and legal teams want the system and design decisions to be well-documented.

Good documentation is necessary not only for engineers, but also legal reasons. During audits by regulatory bodies, documentation that describes the decision-making process, algorithmic process, and data sources must be reviewed. Engineers require good documentation to help with system reproducibility, maintenance, explainability, and future updates. This will allow the system to be easily updated if system performance deteriorates again in the future due to highly increased sophistication of fraud techniques. This documentation requirement can be met by writing down all details at all phases of the design, implementation, and deployment process, compiling and organizing these details in an easily accessible shared location, and having the relevant stakeholders (engineering and legal/compliance teams) review the documentation for completeness.

### 3. CEO and Chief Risk Officer want to prioritize fraud detection rate and maximize true positives.

The main goal of the system is to improve precision, meaning improve instances of fraud detection. This means maximizing true positives, even if the false positive rate is hindered slightly. This is important for brand reputation because customers need to be able to trust the company's ability to accurately detect fradulent transactions, even if this means occaisional notifications asking to verify a flagged transaction. This requirement could be implemented by giving more weight to metrics such as precision and true positive rate when determining system performance.

### 4. Chief Risk Officer wants to minimize the highest risk fraud type, which is account take over.

The highest priority fraud type is to detect account takeover fraud, which involves unauthorized access to a customer's account credentials, which can lead to fraudlent transactions or identity theft. This may lead to multiple fraudulent transactions from one account, causing large financial losses. This requirement can be addressed by testing the model with specific cases of this type of fraud, and ensuring robust performance especially in this type of case.

### 5. Chief Risk Officer wants the system to prioritize reducing the amount of financial loss instead of total number of fraudulent transactions.

According to the Chief Risk Officer, reducing the amount of financial loss caused by fraudlent transactions is more important than reducing the number of fraudlent transactions. Of course, both values must be minimized as much as possible, but if tradeoffs must be made, minimizing financial loss should be prioritized. This requirement may be addressed by using a model that may be able to detect high value fradulent transactions, even if smaller fradulent transactions go undetected. This could involve using more training data from high value fraudulent transactions. This also requires robust testing of high value transactions as well.

### 6. CEO wants new system to exceed previous metrics of 40% precision and 85% recall.

The existing fraud detection system initially had ~40% precision and ~80% recall, so the new and improved system should exceed these metrics. This level of system performance is necessary in order to make the system reliable enough for use in a production environment. This requirement can be achieved by calculating these metrics at each iteration, and fine-tuning the model until the desired performance is reached. To ensure the model will be reliable in production, the training, validation, and test datasets must be represenative of the distribution of real-world data.

### 7. Credit card users want to receive as few transaction confirmation notifications as possible.

User experience and customer satisfaction are important performance indicators for the system. Customer service representatives have shared that customers are frustrated when the current fraud detection system sends them too many alerts regarding flagged transactions. In order to keep customers and maintain a good company reputation and brand loyalty, we must ensure a high quality user experience, which minimizes unnecessary notifications to users. This requirement can be met by conducting user focus groups or surveys to learn how many notifications may be considered reasonable, analyzing any historical data that may associate number of fraud alerts with customer account closures, and conducting user assurance testing prior to system deployment.

### 8. CEO wants fraud to be detected as quickly as possible when the transaction occurs.

In order to maximize the chances of catching the culprit and rectifying the issue in cases of fraud, the fraudlent transactions must be detected as quickly as possible. There have not been any known complaints made regarding the current detection time, but the time to detection will be an important metric to keep track of for the new system. This requirement could be met by maintaining or improving the existing time to detection, testing the new system to get an expected average time to detection, and monitoring this metric in production to ensure the engineering team is alerted in case of any anomolies or increases.

### 9. CEO wants the system to have a flexible and customizable alert system to notify stakeholders such as fraud analysts and customer service representatives about flagged transactions.

A nice feature to have would be a felixble and customizable alert system to notify stakeholders about potentially fraudlent transactions. This alert system could allow internal stakeholders such as fraud analysts and customer service representatives to be of flagged transactions before customers are notified, adding an extra layer of analysis resulting in minimal notifications to the end user. A flexible alert system would allow different stakeholders to receive different types of notifications. For example, an alert sent to a fraud analyst may contain more data than one sent to a customer service representative, which may be important for data access controls. This alert system would add a human element to the automated system, which could decrese time to detection and increase true fraud detection rate. This requirement could be implemented by integrating with a third party messaging or notifcation tool, which will be configured to send specific alerts depending on the automated system output.

### 10. CEO and Head of Credit Card Operations want the system to integrate with customer relationship management platforms, payment systems, analytic tools.

Third party integrations will be an important aspect for improving the internal usability of the fraud detection system and ability to fit it into existing business oeprations and workflows. Integration with customer relationship management platforms will be important to maintain and organize communications with customers regarding fradulent activity, and identifying any trends in fraud frequency for specific customers. Integration with payment systems will be important for getting information from processing transactions and refunding any fraudlent transactions. Finally, integrating with data analytics tools such as datadog, PowerBI, or Tableau to name a couple, will be important for tracking and identifying or forcasting trends in transaction data. This integrations requirement can be implemented by working with the integration specialist or API documentation for each third party product to connect each product to the fraud detection system, and thoroughly testing each connection.
