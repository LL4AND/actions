# **Second Me - Front-end**

The `lpm_frontend` directory contains the front-end portion of the **Second Me** project. It provides the user interface for interacting with the system, allowing users to manage and train their AI, view memory, and more. This module is built with a modern front-end tech stack to ensure a seamless and efficient user experience.

------

## **Overview**

1. **User Interface**
    Provides an intuitive interface for managing Second Me, Upload Memory, and training model.
2. **Real-Time Interaction**
    Seamlessly connects with the back-end for real-time data updates and interactions.
3. **Theme Support**
    Includes customizable themes and network user color identification.

------

## **Tech Stack**

- **Next.js (v14)**: Server-Side Rendering (SSR) and Static Site Generation (SSG).
- **React (v18)**: Component-based UI development.
- **TypeScript**: Strong typing for scalable development.
- **Tailwind CSS**: Utility-first styling with responsive design.
- **Ant Design**: Enterprise-grade UI components.
- **Zustand**: Lightweight state management.
- **Framer Motion** & **Three.js**: Smooth animations and 3D rendering.
- **ESLint** & **Prettier**: Enforced code quality and consistent formatting.

------

## **System Requirements**

- **Node.js**: Version 18.x or higher
- **npm or yarn**: Latest version recommended
- **Browser**: Chrome, Edge, or any modern browser

------

## **Installation and Setup**

### **1. Clone the Repository**

```
git clone https://github.com/mindverse/Second-Me.git
cd Second-Me/lpm_frontend
```

### **2. Install Dependencies**

Using npm:

```
npm install
```

Or using yarn:

```
yarn install
```

### **3. Run the Development Server**

Using npm:

```
npm run dev
```

Or using yarn:

```
yarn dev
```

Once the server is running, open your browser and navigate to:
 [http://localhost:3000](http://localhost:3000/)

------

## **Building for Production**

To build the project for production, run the following command:

```
npm run build
```

Or using yarn:

```
yarn build
```

------

## **Project Structure**

```
src/
  ├── app/                # Pages and routing-related modules
  │   ├── dashboard/
  │   ├── home/
  │   ├── standalone/
  │   ├── globals.css
  │   └── layout.tsx
  │
  ├── components/         # Reusable UI components
  │   ├── AvatarUpload
  │   ├── Celebration
  │   ├── chat
  │   ├── InfoModal
  │   ├── LoadMore
  │   ├── modelConfigModal
  │   ├── ModelStatus
  │   ├── multi-upload-task
  │   ├── NetworkSphere
  │   ├── OnboardingTutorial
  │   ├── playground
  │   ├── roleplay
  │   ├── rooms
  │   ├── SimpleMD
  │   ├── spaces
  │   ├── StatusBar
  │   ├── svgs
  │   ├── train
  │   └── upload
  │
  ├── contexts/           # Global contexts
  │   └── AntdRegistry.tsx
  │
  ├── hooks/              # Custom Hooks
  │   └── useSSE.tsx
  │
  ├── layouts/            # Layout components for page structure
  │   ├── DashboardLayout
  │   └── HeaderLayout
  │
  ├── service/            # API service logic for backend interactions
  │   ├── info.ts
  │   ├── memory.ts
  │   ├── model.ts
  │   ├── modelConfig.ts
  │   ├── role.ts
  │   ├── space.ts
  │   ├── train.ts
  │   └── upload.ts
  │
  ├── store/              # State management tools
  │   ├── useLoadInfoStore.ts
  │   ├── useModelConfigStore.ts
  │   ├── useSpaceStore.ts
  │   ├── useTrainingStore.ts
  │   └── useUploadStore.ts
  │
  ├── types/              # TypeScript type definitions
  │   ├── chat.ts
  │   └── responseModal.ts
  │
  └── utils/              # Utility functions and helpers
      ├── chatStorage.ts
      ├── event.ts
      ├── localRegisteredUpload.ts
      ├── localStorage.ts
      ├── memory.ts
      ├── request.ts
      └── router.ts

```



```
lpm_frontend/
├── public/               # Static assets (images, fonts)
├── rules/eslint/         # ESLint rules configuration
├── src/                  # Source code (components, pages, logic)
├── .eslintignore         # Files and directories ignored by ESLint
├── .eslintrc.js          # ESLint main configuration file
├── .gitignore            # Git ignore rules
├── .prettierignore       # Files and directories ignored by Prettier
├── .prettierrc.js        # Prettier configuration file for code formatting
├── .stylelintignore      # Files and directories ignored by Stylelint
├── .stylelintrc.js       # Stylelint configuration file for CSS/SCSS linting
├── next.config.js        # Next.js configuration file
├── package.json          # Project dependencies and scripts
├── postcss.config.mjs    # PostCSS configuration file
├── tailwind.config.ts    # Tailwind CSS configuration
├── tsconfig.json         # TypeScript configuration file
├── yarn.lock             # Dependency lock file (for Yarn)
└── package-lock.json     # Dependency lock file (for npm)
```

------

## **Scripts**

Here are the most commonly used npm/yarn scripts:

- Start the development server:

  ```
  npm run dev
  ```

- Build for production:

  ```
  npm run build
  ```

- Run code linting:

  ```
  npm run lint
  ```

------

## **Contributing**

We welcome contributions to the `lpm_frontend` project! Here’s how you can contribute:

1. **Fork the repository**.

2. Create a new branch:

   ```
   git checkout -b feature/your-feature-name
   ```

3. Commit your changes and push:

   ```
   git commit -m "Add your message"
   git push origin feature/your-feature-name
   ```

4. Submit a Pull Request.

------



## **License**

This project is licensed under the **Apache 2.0 License**.
