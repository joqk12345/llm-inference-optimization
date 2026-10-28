# 历史启动清单（归档）

> 归档日期：2026-10-29。以下是项目初建时的计划快照，章节、路径、社区、付费服务和时间表均不代表当前状态，不应作为操作指南。当前任务见 [NEXT_STEPS](../../NEXT_STEPS.md)。

## 原始启动计划

This document guides you through the immediate next steps to launch your LLM inference optimization book.

---

## ✅ What's Already Done

You now have a complete repository structure:

```
llm-inference-optimization/
├── README.md                  ✅ Complete project introduction
├── CONTRIBUTING.md            ✅ Contribution guidelines
├── LICENSE                    ✅ MIT License
├── .gitignore                 ✅ Git ignore rules
├── .github/
│   └── ISSUE_TEMPLATE/        ✅ Bug, feature, and question templates
├── chapters/
│   └── chapter02-gpu-basics.md ✅ Complete Chapter 2 (3000+ words)
├── code/
│   └── chapter02/
│       ├── README.md          ✅ Code documentation
│       ├── memory_calculator.py ✅ Working code
│       ├── Dockerfile         ✅ Container definition
│       ├── docker-compose.yml ✅ One-command setup
│       └── requirements.txt   ✅ Dependencies
└── docs/
    ├── faq.md                 ✅ Frequently asked questions
    ├── contributors.md        ✅ Contributor recognition
    └── success-stories.md     ✅ Success story template
```

**Total**: 10 files, ~15,000 words of content

---

## 🚀 Step 1: Create GitHub Repository (5 minutes)

### Option A: Using GitHub CLI (Recommended)

```bash
# Navigate to your project
cd /Users/mac/Documents/workspace/Data/01_Work/Projects/book-craft

# Initialize git
git init
git add .
git commit -m "Initial commit: LLM Inference Optimization book

- Complete README with project overview
- Chapter 2: GPU Basics (3000+ words)
- Code examples with Docker setup
- Contribution guidelines
- Community templates (FAQ, contributors, success stories)"

# Create repository on GitHub
gh repo create llm-inference-optimization \
  --public \
  --description "A practical guide to optimizing LLM inference - from GPU basics to production deployment" \
  --source=. \
  --remote=origin \
  --push

# Your repository is now live at:
# https://github.com/joqk12345/llm-inference-optimization
```

### Option B: Manual GitHub Creation

1. Go to https://github.com/new
2. Repository name: `llm-inference-optimization`
3. Description: `A practical guide to optimizing LLM inference - from GPU basics to production deployment`
4. Set to **Public**
5. Don't initialize with README (we already have one)
6. Click "Create repository"
7. Push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/joqk12345/llm-inference-optimization.git
   git push -u origin main
   ```

---

## 📢 Step 2: Set Up Discord (30 minutes)

### Create Server

1. Go to https://discord.com/new
2. Name: "LLM Inference Optimization"
3. Icon: Upload a book/GPU icon (or use default)

### Create Channels

Create these channels in order:

**General**:
- `#welcome` - Welcome message and rules
- `#announcements` - Important updates (read-only except admins)
- `#rules` - Community guidelines

**Discussion**:
- `#introductions` - Say hi!
- `#questions` - General questions
- `#discussions` - Content discussion
- `#show-off` - Share your projects

**Chapter-Specific**:
- `#chapter01-intro`
- `#chapter02-gpu`
- `#chapter03-environment`
- `#chapter04-kv-cache`
- `#chapter05-scheduling`
- `#chapter06-quantization`
- `#chapter07-speculative`
- `#chapter08-production`
- `#chapter09-advanced`

**Community**:
- `#success-stories` - Share your wins
- `#contributors` - Contributor recognition
- `#office-hour` - Q&A session info
- `#off-topic` - Casual chat

### Roles to Create

- **@Everyone** - Default permissions
- **@Contributor** - Anyone who contributes (gets special color)
- **@Top Contributor** - Monthly top 10 (gets early access)
- **@Moderator** - Community helpers
- **@Admin** - You

### Welcome Message Template

```markdown
# Welcome to LLM Inference Optimization! 🚀

This community is for readers of the open-source book "LLM Inference Optimization".

## 📖 About the Book
- Free, practical guide to optimizing LLM inference
- 90% content open source
- Real code examples with Docker
- No deep learning background required

## 🎯 What to Do First
1. Introduce yourself in #introductions
2. Read the rules in #rules
3. Check out the book: https://github.com/joqk12345/llm-inference-optimization
4. Join the discussion for your chapter

## 📅 Office Hours
- Wednesdays 20:00-21:00 UTC
- Fridays 20:00-21:00 UTC
- In #office-hour

## 🏆 Contributing
We welcome all contributions! See CONTRIBUTING.md in the repo.

Top monthly contributors get free Pro access!

Ask questions, share projects, help others learn. Let's optimize! ⚡
```

### Invite Link

Create a permanent invite link:
1. Server Settings → Invites
2. Create instant invite
3. Set to never expire
4. Copy and add to README.md

---

## 🎥 Step 3: Record Introduction Video (1-2 hours)

### Equipment Needed
- **Camera**: Phone or webcam (1080p)
- **Microphone**: Lapel mic or decent USB mic
- **Lighting**: Natural light or ring light
- **Background**: Clean, uncluttered

### Recording Tips
1. **Script**: Use the 5-minute script from the plan
2. **Practice**: Do 2-3 practice runs
3. **Environment**: Quiet room, good lighting
4. **Screen Recording**: Include code demos
5. **Length**: Keep it under 5 minutes

### Editing (Free Options)
- **DaVinci Resolve** (https://www.blackmagicdesign.com/products/davinciresolve)
- **iMovie** (Mac only)
- **Clipchamp** (Windows 11)

### Must-Have Content
- ✅ Introduction (who you are)
- ✅ Problem (why this book matters)
- ✅ Solution (what the book teaches)
- ✅ Call to action (star the repo, join Discord)

### Upload
1. Create YouTube channel
2. Upload video
3. Add to README.md:
   ```markdown
   [![Watch the video](thumbnail.jpg)](https://youtube.com/watch?v=TODO)
   ```

---

## 📢 Step 4: Launch Day Promotion (2 hours)

### Pre-Launch Checklist
- [ ] Repository is public
- [ ] README looks good
- [ ] Discord server is ready
- [ ] Video is uploaded
- [ ] You have the GitHub URL and Discord invite handy

### Launch Platforms

**1. HackerNews (Show HN)**
- URL: https://news.ycombinator.com/item?id=TODO
- Title: "Show HN: I'm writing an open-source book on LLM inference optimization"
- Post when: Tuesday-Thursday, 8-11 AM US Eastern
- Template:
  ```markdown
  Hi HN,

  I'm writing a practical guide to LLM inference optimization.

  Why: ChatGPT is fast, but running LLMs locally is slow. I spent a year
  researching vLLM and optimization techniques. Now I'm sharing what I learned.

  What: Free book covering GPU basics, KV cache, scheduling, quantization,
  speculative sampling, and production deployment.

  Target: Developers who know Python but no deep learning background required.

  Status: Just launched with Chapter 2 complete. Planning to finish all 9 chapters
  in 5 months.

  90% will be free. The book includes code examples with Docker setup.

  GitHub: https://github.com/joqk12345/llm-inference-optimization

  Feedback welcome!
  ```

**2. Reddit**
- r/MachineLearning: https://reddit.com/r/MachineLearning/submit
- r/LocalLLaMA: https://reddit.com/r/LocalLLaMA/submit
- Title: "[D] Open-source book: LLM Inference Optimization - Practical guide from GPU basics to production"
- Similar template to HackerNews

**3. Twitter/X**
```markdown
🚀 I'm writing an open-source book on LLM Inference Optimization!

Learn to optimize LLMs like a pro:
✅ GPU basics & architecture
✅ KV Cache, scheduling, quantization
✅ Production deployment
✅ Complete code with Docker

90% free, no DL background required

GitHub: https://github.com/joqk12345/llm-inference-optimization

First 100 stars get special recognition! ⭐

#LLM #MachineLearning #OpenSource
```

**4. LinkedIn**
- More professional tone
- Focus on practical applications
- Mention the 5-month timeline

**5. Chinese Communities**
- 知乎: Create a专栏
- 掘金: Technical article
- V2EX: Share in dev community

---

## 📊 Step 5: Monitor and Engage (First Week)

### Day 1-3: Launch Phase
- Respond to every comment on HN/Reddit
- Welcome every Discord member
- Fix immediate issues
- Track metrics

### Key Metrics to Watch
- GitHub Stars: Aim for 100 in first week
- Discord Members: Aim for 50 in first week
- Video Views: Nice to have, not critical
- Issues/PRs: First contributions!

### Day 4-7: Follow-up
- Send thank you to early contributors
- Post progress update
- Ask for feedback
- Plan next chapter based on interest

---

## 🎯 Success Criteria (Week 1)

If you achieve these by Day 7, you're on track:
- ✅ 50+ GitHub stars
- ✅ 30+ Discord members
- ✅ At least one community contribution
- ✅ Chapter 3 outline started
- ✅ Scheduled first Office Hour

---

## 📅 Week 2 Preview

Once Week 1 is done:
1. Start writing Chapter 1 (Introduction)
2. Create content calendar
3. Set up analytics (optional)
4. Schedule regular Office Hours

---

## ❓ Need Help?

### Technical Issues
- GitHub: Check their documentation
- Discord: Check their server setup guide
- Video editing: YouTube tutorials

### Content Questions
- Review the full plan: `docs/plans/2025-01-26-llm-inference-book-plan.md`
- Ask in Discord once it's set up

### Motivation
- Remember:的影响力优先，收入自然跟随
- You've already done the hard work (planning)
- Now just execute!

---

## 🚀 You're Ready!

Everything you need is in place. The repository is complete, the plan is solid, and the path is clear.

**Your next action**: Create the GitHub repository and push the code.

**Good luck! You've got this!** 💪

---

*Created: 2025-01-26*
*Last updated: 2025-01-26*
