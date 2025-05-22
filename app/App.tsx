import React from 'react';
import AppTabs from './(tabs)/_layout';
import { UserProvider } from './userContext';

export default function App() {
  return (
    <UserProvider>
      <AppTabs />
    </UserProvider>
  );
}
