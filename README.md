\# 🚀 Blockflow Exchange



\*\*Version:\*\* 3.5.1 (Investor-Ready)  

\*\*Status:\*\* Production-Grade Prototype  

\*\*Target:\*\* India-First Cryptocurrency Exchange with Auto-TDS Compliance



---



\## 🎯 What is Blockflow?



Blockflow is a \*\*compliance-first cryptocurrency exchange\*\* designed specifically for the Indian market. Our key differentiation is \*\*automated TDS (Tax Deducted at Source) calculation\*\* per Section 194S of the Income Tax Act, addressing the #1 pain point for Indian crypto traders.



\### Key Features



\- ✅ \*\*Real Order Matching\*\* - FIFO limit order book with partial fills

\- ✅ \*\*Auto-TDS Calculation\*\* - 1% automatic tax deduction with Form 26QE support

\- ✅ \*\*Balance Management\*\* - Locked/available balance tracking

\- ✅ \*\*Real-time WebSocket\*\* - Live price feeds and order updates

\- ✅ \*\*Audit Trail\*\* - Complete compliance-grade logging with correlation IDs

\- ✅ \*\*Demo/Live Toggle\*\* - Clearly separated testing and production modes

\- ✅ \*\*NO FAKE DATA\*\* - All metrics are real from PostgreSQL



---



\## 🏗️ Tech Stack



\- \*\*Backend:\*\* FastAPI 0.104+ (Python 3.11)

\- \*\*Database:\*\* PostgreSQL 15+ (with Alembic migrations)

\- \*\*Caching:\*\* Redis 7+ (optional, for multi-instance)

\- \*\*Auth:\*\* JWT (HS256) + Bcrypt password hashing

\- \*\*WebSocket:\*\* Native FastAPI WebSocket support

\- \*\*Deployment:\*\* Docker + Docker Compose



---



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.11+

\- PostgreSQL 15+ (or use Docker)

\- Redis 7+ (optional)



\### 1. Clone \& Setup

```bash

git clone https://github.com/yourorg/blockflow.git

cd blockflow



\# Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate



\# Install dependencies

pip install -r requirements

