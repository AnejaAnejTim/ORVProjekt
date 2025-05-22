import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import React, { useContext } from 'react';
import Home from './(tabs)/index';
import Profile from './(tabs)/profile';
import Login from './login';
import { UserContext } from './userContext';

const Tab = createBottomTabNavigator();

function AppTabs() {
  const context = useContext(UserContext);

  if (!context) {
    throw new Error("UserContext must be used within a UserProvider");
  }

  const { user } = context;

  return (
    <Tab.Navigator>
      {user ? (
        <>
          <Tab.Screen name="Home" component={Home} />
          <Tab.Screen name="Profile" component={Profile} />
        </>
      ) : (
        <>
          <Tab.Screen
            name="Login"
            component={Login}
            options={{ tabBarButton: () => null }}
          />
        </>
      )}
    </Tab.Navigator>
  );
}

export default AppTabs;
