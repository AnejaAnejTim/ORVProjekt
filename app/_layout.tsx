import { Slot } from 'expo-router';
import React, { useContext } from 'react';
import { Text, View } from 'react-native';
import { UserContext, UserProvider } from './userContext';

function InnerLayout() {
  const context = useContext(UserContext);
  if (!context) return null;

  const { loading } = context;

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <Text>Loading user...</Text>
      </View>
    );
  }

  return <Slot />;
}

export default function RootLayout() {
  return (
    <UserProvider>
      <InnerLayout />
    </UserProvider>
  );
}
