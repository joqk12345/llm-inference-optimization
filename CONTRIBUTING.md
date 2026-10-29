# Contributing to LLM Inference Optimization

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🤝 How to Contribute

There are many ways to contribute, and all of them are valuable:

### 📝 Content Contributions

#### Reporting Issues
If you find an error in the content:
1. Check if an issue already exists
2. Use the appropriate issue template (Bug Report, Content Error, or Question)
3. Provide as much detail as possible:
   - Chapter and section
   - The error or confusion
   - Suggested correction (if you have one)

#### Suggesting Improvements
Have an idea for improving the content?
1. Open an issue with the "enhancement" label
2. Describe what you'd like to see added or changed
3. Explain why it would be valuable

#### Writing Content
Want to contribute actual content?
1. Start with a discussion or issue to propose your idea
2. Wait for feedback before writing
3. Follow the existing style and structure
4. Submit a pull request with your changes

### 💻 Code Contributions

#### Fixing Bugs
Found a bug in the code examples?
1. Fork the repository
2. Create a branch for your fix
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

#### Adding Examples
Want to add more examples?
1. Check if the example fits in an existing chapter
2. Follow the code structure:
   ```
   code/chapterXX/
   ├── README.md           # Explanation
   ├── example.py          # Main code
   ├── Dockerfile          # Container definition
   └── requirements.txt    # Dependencies
   ```
3. Test it works with Docker
4. Submit a pull request

#### Improving Documentation
Documentation improvements are always welcome:
- Better comments in code
- More detailed explanations
- Additional use cases
- Performance benchmarks

### 📖 Success Stories

Share your experience applying the techniques from this book:

1. **Format**:
   ```markdown
   ## Success Story: [Title]

   **Reader**: [Your Name/Pseudonym]
   **Role**: [Your Role]
   **Problem**: [What issue were you facing?]
   **Solution**: [Which techniques from the book did you apply?]
   **Results**: [Quantitative improvements if possible]

   **Key Learnings**:
   - Learning 1
   - Learning 2
   ```

2. **Where to submit**:
   - GitHub Issue with the "success-story" label
   - Include reproducible commands, environment details and raw results; label hypothetical examples clearly.

### 💬 Community Contributions

#### Answering Questions
Help others in the community:
- Monitor GitHub Issues and Discussions
- Include chapter and section references when answering
- Provide helpful, accurate answers

#### Organizing Events
Want to host a study group or discussion?
- Propose the event in a GitHub issue
- We can help promote and organize

#### Sharing on Social Media
Help spread the word:
- Share your progress
- Post about interesting insights
- Tag the project

## 📋 Contributor Recognition

We value all contributions and recognize our contributors:

Contributions are visible in GitHub history. Contributor notes are maintained in [docs/contributors.md](docs/contributors.md). This repository does not currently publish a Pro membership, consulting reward, or recurring event program.

## 🎯 Contribution Areas

We're currently looking for help in:

### High Priority
- 📝 Content review and feedback
- 🐛 Bug reports for code examples
- 💡 Ideas for practical examples
- 📖 Success stories from early readers

### Medium Priority
- 🌐 Translation to other languages
- 🎨 Visualizations and diagrams
- 📊 Performance benchmarks
- 🧪 Test cases for code examples

### Low Priority
- 🎥 Video production help
- 💬 Community moderation
- 📢 Social media management
- 🎨 Design improvements

## 📝 Style Guidelines

### Writing Style
- **Tone**: Practical, conversational, and clear
- **Audience**: Developers with Python knowledge
- **Format**: Markdown with clear sections
- **Code**: Well-commented, following PEP 8

### Code Style
- Use type hints where appropriate
- Include docstrings for functions
- Add inline comments for complex logic
- Test with Docker before submitting

### Common Patterns
- **Code blocks**: Use triple backticks with language specifier
- **Commands**: Use bash code blocks
- **Emphasis**: Use bold for key terms, italic for emphasis
- **Lists**: Use numbered lists for steps, bullet points for options

## 🔄 Pull Request Process

1. **Before You Start**
   - Check if an issue or discussion already exists
   - Comment on the issue to claim it (if applicable)
   - Wait for confirmation if it's a major change

2. **Making Changes**
   - Create a descriptive branch name: `fix/chapter03-gpu-math-error`
   - Make your changes
   - Test thoroughly
   - Update documentation if needed

3. **Submitting PR**
   - Fill out the PR template
   - Link to related issues
   - Describe your changes clearly
   - Request review from maintainers

4. **After Submission**
   - Respond to review feedback
   - Make requested changes
   - Be patient - maintainers are volunteers!

## 🚫 Code of Conduct

### Our Pledge
We are committed to making this community a harassment-free experience for everyone.

### Our Standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Gracefully accept constructive criticism
- Focus on what is best for the community

### Unacceptable Behavior
- Harassment or discriminatory language
- Personal attacks or insults
- Public or private harassment
- Publishing others' private information
- Unprofessional conduct

### Enforcement
Violations will be addressed by the maintainers and may result in:
- Warning
- Temporary ban
- Permanent ban

## 📧 Getting Help

Have questions about contributing?
- Open a discussion with the "question" label
- Include the chapter, environment and reproduction steps in your issue

## 🙏 Thank You

Whether you're fixing a typo, writing a whole chapter, or just spreading the word - **we appreciate you!**

This project wouldn't be possible without contributors like you.

---

**Ready to contribute? Check out our [good first issues](https://github.com/joqk12345/llm-inference-optimization/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)!** 🚀
