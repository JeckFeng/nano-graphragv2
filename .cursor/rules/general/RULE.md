---
alwaysApply: true
---

# python编程通用规则:
    description: Python通用编码规范
    globs: ["**/*.py"]
    instructions: |
      - 遵循PEP 8Python代码风格指南
      - 使用Python 3.10及以上的语法特性和最佳实践
      - 合理使用面向对象编程(00P)和函数式编程范式
      - 使用类型注解(Type Hints)
      - 使用类型提示(Type Hints)进行类型检查，提高代码质量
      - 编写详细的docstrings文档
      - 使用描述性的变量和函数命名
      - 实现优雅的错误处理机制
      - 编写详细的文档字符串(docstring)和注释；模块 docstring 放在文件顶部，简要说明用途、边界、设计约束。
      - 每个函数都必须添加函数注释（docstring + 类型注解）
      - 每个类都必须类 添加docstring 说明 职责、重要不变量（invariants）
      - 实现适当的错误处理和日志记录，按需编写单元测试确保代码质量

# 代码风格
    description: 代码风格
    globs: ["**/*.*"]
    instructions: |
      - 保持代码简洁、可读
      - 使用有意义的变量和函数名
      - 添加适当的注释解释复杂逻辑
      - 遵循每种语言的官方风格指南

# 项目结构
    description: 项目结构
    globs: ["**/*.*"]
    instructions: |
      - 保持项目结构清晰，遵循模块化原则
      - 相关功能应放在同一目录下
      - 使用适当的目录命名，反映其包含内容

# 通用开发原则
    description: 通用开发原则
    globs: ["**/*.*"]
    instructions: |
      - 编写可测试的代码
      - 避免重复代码(DRY原则)
      - 优先使用现有库和工具，避免重新发明轮子
      - 考虑代码的可维护性和可扩展性

# 响应语言
    - 始终用中文回答