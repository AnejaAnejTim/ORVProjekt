import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { ReactNode, useEffect, useState } from 'react';
import { User } from './types';

interface UserContextType {
  user: User | null;
  setUser: React.Dispatch<React.SetStateAction<User | null>>;
  loading: boolean;
}

export const UserContext = React.createContext<
  UserContextType & { refreshUser: () => Promise<void> } | undefined
>(undefined);

export const UserProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = async () => {
    setLoading(true);
    try {
      const token = await AsyncStorage.getItem('token');
      console.log('refreshUser: token from AsyncStorage:', token);

      if (!token) {
        console.log('No token found, setting user to null');
        setUser(null);
        setLoading(false);
        return;
      }
      const response = await fetch(`http://192.168.0.13:3001/users/appValidation`, {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      console.log('refreshUser: appValidation response status:', response.status);

      if (response.ok) {
        const profile = await response.json();
        console.log('refreshUser: user profile fetched:', profile);
        await AsyncStorage.setItem('userId', profile._id);
        setUser(profile);
      } else {
        console.log('refreshUser: invalid token, removing token and clearing user');
        setUser(null);
        await AsyncStorage.removeItem('token');
        await AsyncStorage.removeItem('userId');
      }
    } catch (error) {
      console.error('Failed to load or validate user', error);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshUser();
  }, []);

  useEffect(() => {
    if (!loading && !user) {
      AsyncStorage.removeItem('token').catch(console.error);
    }
  }, [user, loading]);

  return (
    <UserContext.Provider value={{ user, setUser, loading, refreshUser }}>
      {children}
    </UserContext.Provider>
  );
};
