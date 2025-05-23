import AsyncStorage from '@react-native-async-storage/async-storage';
import React from 'react';
import AppTabs from './(tabs)/_layout';
import { UserProvider } from './userContext';

const originalSetItem = AsyncStorage.setItem;
AsyncStorage.setItem = async (key, value) => {
  console.log(`AsyncStorage.setItem key=${key}, value=${value}`);
  return originalSetItem(key, value);
};

const originalRemoveItem = AsyncStorage.removeItem;
AsyncStorage.removeItem = async (key) => {
  console.log(`AsyncStorage.removeItem key=${key}`);
  return originalRemoveItem(key);
};


export default function App() {
  return (
    <UserProvider>
      <AppTabs />
    </UserProvider>
  );
}
